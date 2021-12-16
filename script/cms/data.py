"""CMS/data"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from script import utils
from script.datasets import Dataset

from typing import Union


def sample(df: pd.DataFrame, amount: int, seed: Union[int, np.random.RandomState]):
    """Samples a random `amount` of events from given pd.DataFrame `df`"""
    amount = int(amount)

    if amount > df.shape[0]:
        x = []
        
        while amount > 0:
            x.append(df.sample(n=min(amount, df.shape[0]), random_state=seed))
            amount -= df.shape[0]
        
        return pd.concat(x, axis=0)
    
    return df.sample(n=amount, random_state=seed)


def np_sample(x: np.ndarray, amount: int, generator):
    """Samples a random `amount` of events from given np.ndarray `x`"""
    amount = int(amount)

    indices = generator.choice(np.arange(x.shape[0]), size=amount, replace=True)
    return x[indices]


def train_val_test_split(dataset: Dataset, valid_size=0.25, test_size=0.2, seed=utils.SEED):
    assert isinstance(dataset, Dataset)

    sig = dataset.signal
    bkg = dataset.background
    
    # test split
    train_sig, test_sig = train_test_split(sig, test_size=test_size, random_state=seed)
    train_bkg, test_bkg = train_test_split(bkg, test_size=test_size, random_state=seed)
    
    # train-valid split
    train_sig, valid_sig = train_test_split(train_sig, test_size=valid_size, random_state=seed)
    train_bkg, valid_bkg = train_test_split(train_bkg, test_size=valid_size, random_state=seed)
    
    return (train_sig, train_bkg), (valid_sig, valid_bkg), (test_sig, test_bkg)


def train_valid_split(dataset: Dataset, valid_size=0.25, seed=utils.SEED):
    assert isinstance(dataset, Dataset)

    sig = dataset.signal
    bkg = dataset.background
    
    # validation split
    train_sig, valid_sig = train_test_split(sig, test_size=valid_size, random_state=seed)
    train_bkg, valid_bkg = train_test_split(bkg, test_size=valid_size, random_state=seed)
    
    return (train_sig, train_bkg), (valid_sig, valid_bkg)


class BaselineSequence(tf.keras.utils.Sequence):
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, delta=50, balance_signal=True, seed=utils.SEED):
        self.sig = signal[features + ['mA', 'type']]
        self.bkg = background[features + ['mA', 'type']].values
        
        self.mass = sorted(self.sig['mA'].unique())
        self.gen = utils.get_random_generator(seed)   # "fast" random generator for `np.random.choice`

        self.should_balance = bool(balance_signal)
        self.bkg_batch = batch_size // 2
        
        if self.should_balance:
            self.sig_batch = (batch_size / 2) // len(self.signals.keys())
            self.signals = {m: self.sig[self.sig['mA'] == m].values for m in self.mass}
        else:
            self.sig = self.sig.values
            self.sig_batch = batch_size // 2
    
    def __len__(self):
        return self.sig.shape[0] // self.bkg_batch  # bkg_batch = half batch-size
    
    def __getitem__(self, idx):
        if self.should_balance:
            z = [np_sample(sig, amount=self.sig_batch, generator=self.gen) for sig in self.signals.values()]
        else:
            z = [np_sample(self.sig, amount=self.sig_batch, generator=self.gen)]
        
        z.append(np_sample(self.bkg, amount=self.bkg_batch, generator=self.gen))
        
        z = np.concatenate(z, axis=0)
        
        x = z[:, :-2]
        m = z[:, -2]
        y = z[:, -1]
        
        # sample mass (from signal's mA) for background events
        mask = y == 0.0
        m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True)

        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))
        
        return dict(x=x, m=m), y

    @classmethod
    def get_data(cls, dataset: Dataset, features: list, train_batch=128, eval_batch=1024, **kwargs):
        # split data
        train, valid, test = train_val_test_split(dataset, **kwargs)
        
        # create sequences
        train_seq = cls(signal=train[0], background=train[1], batch_size=train_batch, features=features)

        valid_seq = cls(signal=valid[0], background=valid[1], batch_size=eval_batch,
                        features=features, balance_signal=False)

        test_seq = cls(signal=test[0], background=test[1], batch_size=eval_batch, 
                       features=features, balance_signal=False)
        
        # create tf.Datasets
        train_ds = utils.dataset_from_sequence(train_seq)
        valid_ds = utils.dataset_from_sequence(valid_seq)
        test_ds = utils.dataset_from_sequence(test_seq)
        
        return train_ds, valid_ds, test_ds


class BalancedSequence(tf.keras.utils.Sequence):
    """keras.Sequence that balances the signal (each mA has the same number of events), and backgrounds"""

    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, balance_signal=True, balance_bkg=True, seed=utils.SEED,
                 sample_mass=False, bins: list = None):
        # select data
        self.sig = signal[features + ['mA', 'type', 'dimuon_M']]
        self.names = background['name']
        self.bkg = background[features + ['mA', 'type', 'dimuon_M']]
        
        self.mass = np.sort(self.sig['mA'].unique())

        if bins is None:
            self.mass_intervals = np.array(Dataset.MASS_INTERVALS)
            self.mass_intervals = self.mass_intervals[:, 1]  # = disjoint bins

            self.binned_mass = np.mean(Dataset.MASS_INTERVALS, axis=-1).reshape(-1, 1)
        else:
            self.mass_intervals = np.array(bins)

            # considered bins (mass_intervals) may contain more than one mass; so sample them, if so
            buckets = []

            for low, up in zip(self.mass_intervals[:-1], self.mass_intervals[1:]):
                bucket = []
                
                for mass in self.mass:
                    if mass > low and mass <= up:
                        bucket.append(mass)
                
                buckets.append(bucket)

            self.binned_mass = np.array(buckets)            

        self.should_sample_mass = bool(sample_mass)
        
        self.gen = utils.get_random_generator(seed)   # "fast" random generator for `np.random.choice`
        
        self.should_balance_sig = bool(balance_signal)
        self.should_balance_bkg = bool(balance_bkg)
        
        # self.features = features
        self.half_batch = batch_size // 2
        
        if self.should_balance_sig:
            self.signals = {m: self.sig[self.sig['mA'] == m].values for m in self.mass}
            
            self.sig_batch = self.half_batch // len(self.signals.keys())
        else:
            self.sig_batch = self.half_batch
            self.sig = self.sig.values
            
        if self.should_balance_bkg:
            self.bkgs = {k: self.bkg[self.names == k].values for k in self.names.unique()}
            
            self.bkg_batch = self.half_batch // len(self.bkgs.keys())
        else:
            self.bkg_batch = self.half_batch
            self.bkg = self.bkg.values
    
    def __len__(self):
        return self.sig.shape[0] // self.half_batch
    
    def __getitem__(self, idx):
        if self.should_balance_sig:
            z = [np_sample(sig, amount=self.sig_batch, generator=self.gen) for sig in self.signals.values()]
        else:
            z = [np_sample(self.sig, amount=self.half_batch, generator=self.gen)]
        
        if self.should_balance_bkg:
            z.extend([np_sample(bkg, amount=self.bkg_batch, generator=self.gen) for bkg in self.bkgs.values()])
        else:
            z.append(np_sample(self.bkg, amount=self.half_batch, generator=self.gen))
        
        z = np.concatenate(z, axis=0)

        # split data
        x = z[:, :-3]
        m = z[:, -3]
        y = z[:, -2]
        
        if self.should_sample_mass:
            # sample mass (from signal's mA) for background events -> uses all background events
            mask = y == 0.0
            m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True)
        else:
            # take mass from corresponding mass interval (m - delta[i], m + delta[i])
            mask = y == 0.0
            dimuon_mass = z[:, -1]
            
            idx = np.digitize(dimuon_mass[mask], self.mass_intervals, right=True)
            # idx = np.clip(idx, a_min=0, a_max=len(self.mass) - 1)
            idx = np.clip(idx, a_min=0, a_max=len(self.binned_mass) - 1)

            # m[mask] = self.mass[idx]
            m[mask] = [self.gen.choice(bucket) for bucket in self.binned_mass[idx]]
    
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        return dict(x=x, m=m), y

    @classmethod
    def get_data(cls, dataset: Dataset, features: list, case: int, train_batch=128, eval_batch=1024, bins=None,
                 **kwargs):
        # split data
        train, valid, test = train_val_test_split(dataset, **kwargs)
        
        # create sequences
        train_seq = cls(signal=train[0], background=train[1], batch_size=train_batch, 
                        features=features, sample_mass=case == 1, bins=bins)

        valid_seq = cls(signal=valid[0], background=valid[1], batch_size=eval_batch, bins=bins,
                        features=features, balance_signal=False, balance_bkg=False, sample_mass=False)

        test_seq = cls(signal=test[0], background=test[1], batch_size=eval_batch, bins=bins,
                       features=features, balance_signal=False, balance_bkg=False, sample_mass=False)
        
        # create tf.Datasets
        train_ds = utils.dataset_from_sequence(train_seq)
        valid_ds = utils.dataset_from_sequence(valid_seq)
        test_ds = utils.dataset_from_sequence(test_seq)
        
        return train_ds, valid_ds, test_ds


class MassBalancedSequence(tf.keras.utils.Sequence):
    """tf.keras.Sequence with balanced batches for each mass"""
    
    # TODO: normalize mA e.g. divide by 1000?
    # TODO: normalize features?
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, features: list,
                 beta=0.5, delta=50, balance=True, weight_column=None, sample_mass=False,
                 seed=utils.SEED, intervals=None):
        columns = features + ['mA', 'type', 'dimuon_M']
        self.indices = {'features': -3, 'mass': -3, 'label': -2, 'dimuon_M': -1}
        
        sig = signal
        bkg = background
        
        if isinstance(weight_column, str):
            assert (weight_column in sig) and (weight_column in bkg)
            self.should_weight = True

            columns += [weight_column]

            self.indices = {k: idx - 1 for k, idx in self.indices.items()}  # decrease index by "-1"
            self.indices['weight'] = -1  # add new index for sample-weights
        else:
            self.should_weight = False

        self.mass = np.sort(sig['mA'].unique())

        if intervals is None:
            # use the "mass" to determine the intervals given "deltas"
            if isinstance(delta, (int, float)):
                self.deltas = [(delta, delta)] * len(self.mass)

            elif isinstance(delta, tuple):
                self.deltas = [delta] * len(self.mass)
            else:
                assert isinstance(delta, (list, np.ndarray))
                assert len(delta) == len(self.mass)

                self.deltas = delta

            self.deltas = np.array(self.deltas)
            
            self.mass_intervals = np.stack([self.mass - self.deltas[:, 0],
                                            self.mass + self.deltas[:, 1]], axis=-1)
        else:
            self.mass_intervals = np.array(intervals)
            self.deltas = np.abs(self.mass_intervals - self.mass[:, np.newaxis])

        # considered bins (mass_intervals) may contain more than one mass; so sample them, if so
        buckets = []

        for low, up in zip(self.mass_intervals[:, 0], self.mass_intervals[:, 1]):
            bucket = []
            
            for mass in self.mass:
                if mass >= low and mass <= up:
                    bucket.append(mass)
            
            buckets.append(bucket)

        self.binned_mass = np.array(buckets)

        # make sure to take all the remaining events
        # self.deltas[0] = (np.inf, self.deltas[0][1])
        # self.deltas[-1] = (self.deltas[-1][0], np.inf)  

        self.deltas = self.deltas.astype(np.float32)

        self.deltas[0][0] = np.inf
        self.deltas[-1][1] = np.inf  

        self.should_balance = bool(balance)
        self.should_sample_mass = bool(sample_mass)
        
        self.gen = utils.get_random_generator(seed)
        self.num_batches = (sig.shape[0] + bkg.shape[0]) // batch_size
        
        if self.should_balance:
            self.batch_sig = (batch_size / 2) // self.mass.shape[0]
            self.batch_bkg = ((batch_size / 2) / self.mass.shape[0]) // len(bkg['name'].unique()) 
            
            self.signals = {}
            self.bkgs = {}
            
            s_ = {m: sig[sig['mA'] == m] for m in self.mass}
            b_ = {m: bkg[(bkg['dimuon_M'] > m - d1) & (bkg['dimuon_M'] < m + d2)] for m, (d1, d2) in zip(self.mass, self.deltas)}
        
            # # num. sig and bkg such that #sig ~ #bkg
            # num_s = list(map(lambda x: x.shape[0], s_.values()))
            # num_b = list(map(lambda x: x.shape[0], b_.values()))
            
            # # compute weights
            # ws = []
            # wb = []
            
            # slice data
            for i, m in enumerate(self.mass):
                s = s_[m]
                b = b_[m]
                
                # w_s = ws[i]
                # w_b = wb[i]
                
                # signal
                # w_s = np.ones((num_s[i], 1)) * w_s
                # self.signals[m] = np.concatenate([s[columns].values, w_s], axis=-1)
                self.signals[m] = s[columns].values
                
                # background
                self.bkgs[m] = {}
                
                for name in b['name'].unique():
                    # b_name = b[b['name'] == name]
                    # w_name = np.ones((b_name[weight_column].shape[0], 1)) * w_b
                    
                    # self.bkgs[m][name] = np.concatenate([b_name[columns].values, w_name], axis=-1)
                    self.bkgs[m][name] = b[b['name'] == name][columns].values
        else:
            s = sig[columns]
            b = bkg[columns]
            
            self.all = np.concatenate([s.values, b.values], axis=0)
            self.batch_size = int(batch_size)
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        if self.should_balance:
            # sample signal (each mA)
            z = [np_sample(s, amount=self.batch_sig, generator=self.gen) for s in self.signals.values()]
            
            # sample background (each mA and process)
            m_bkg = []

            for m, bkg in self.bkgs.items():
                for b in bkg.values():
                    z.append(np_sample(b, amount=self.batch_bkg, generator=self.gen))

                    m_bkg.append(m * np.ones((z[-1].shape[0],)))

            m_bkg = np.concatenate(m_bkg, axis=0)
        else:
            z = [np_sample(self.all, amount=self.batch_size, generator=self.gen)]
        
        z = np.concatenate(z, axis=0)
        
        # split data (features, mass, label, dimuon_M, weight)
        x = z[:, :self.indices['features']]
        m = z[:, self.indices['mass']]
        y = z[:, self.indices['label']]
        
        if self.should_sample_mass:
            # sample mass (from signal's mA) for background events -> uses all background events
            mask = y == 0.0
            m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True)
        else:
            # take mass from corresponding mass interval
            mask = y == 0.0
            # dimuon_mass = z[:, self.indices['dimuon_M']]
            
            # idx = np.digitize(dimuon_mass[mask], self.mass_intervals, right=True)
            # # idx = np.clip(idx, a_min=0, a_max=len(self.mass) - 1)
            # idx = np.clip(idx, a_min=0, a_max=self.binned_mass.shape[0] - 1)
    
            # # m[mask] = self.mass[idx]
            # m[mask] = [self.gen.choice(bucket) for bucket in self.binned_mass[idx]]

            if self.should_balance:
                m[mask] = m_bkg
            else:
                dimuon_mass = z[:, self.indices['dimuon_M']]

                idx = np.digitize(dimuon_mass[mask], self.mass_intervals[:, 1], right=True)
                idx = np.clip(idx, a_min=0, a_max=len(self.mass) - 1)
        
                m[mask] = [self.gen.choice(bucket) for bucket in self.binned_mass[idx]]
    
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        if self.should_weight:
            w = z[:, self.indices['weight']]
            w = np.reshape(w, newshape=(-1, 1))
            
            return dict(x=x, m=m), y, w
        
        return dict(x=x, m=m), y

    def to_tf_dataset(self):
        return utils.dataset_from_sequence(self, sample_weights=self.should_weight)

    @classmethod
    def get_data(cls, dataset: Dataset, case: int, train_batch=128, eval_batch=1024, 
                 num_splits=2, **kwargs):
        assert num_splits in [2, 3]

        # split data
        if num_splits == 2:
            splits = train_valid_split(dataset)
        else:
            splits = train_val_test_split(dataset)
        
        # create sequences
        sequences = []

        for i, (sig, bkg) in enumerate(splits):
            if i == 0:
                batch_size = train_batch
                sample_mass = case == 1
                balance = True
            else:
                batch_size = eval_batch
                sample_mass = False
                balance = False

            seq = cls(signal=sig, background=bkg, batch_size=batch_size, sample_mass=sample_mass,
                      features=dataset.columns['feature'], balance=balance, **kwargs)

            sequences.append(seq)
        
        # return tf.Datasets
        return [seq.to_tf_dataset() for seq in sequences]


class SingleBalancedSequence(tf.keras.utils.Sequence):
    """Balanced tf.keras.Sequence for individual NNs trained only on one mass"""

    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, mass: float, batch_size: int, 
                 case: int, features: list, delta=50, balance=True, seed=utils.SEED):
        
        self.should_balance = bool(balance)
        
        sig = signal[signal['mA'] == mass]
        bkg = background
        
        # case
        if case == 1:
            # random flat background
            pass
        else:
            # case 2: bkg centered around mass in dimuon_M
            bkg = bkg[(bkg['dimuon_M'] > mass - delta) & (bkg['dimuon_M'] < mass + delta)]
        
        self.sig = sig[features + ['mA', 'type']]
        self.names = bkg['name']
        self.bkg = bkg[features + ['mA', 'type']]

        self.gen = utils.get_random_generator(seed)
        self.mass = float(mass)
        self.sig_batch = batch_size // 2

        if self.should_balance:
            self.bkg = {k: self.bkg[self.names == k].values for k in self.names.unique()}
            self.bkg_batch = self.sig_batch // len(self.bkg.keys())
        else:
            self.bkg_batch = batch_size // 2
    
    def __len__(self):
        return self.sig.shape[0] // self.sig_batch
    
    def __getitem__(self, idx):
        if self.should_balance:
            z = [np_sample(bkg, amount=self.bkg_batch, generator=self.gen) for bkg in self.bkg.values()]
        else:
            z = [np_sample(self.bkg, amount=self.bkg_batch, generator=self.gen)]
        
        z.append(np_sample(self.sig, amount=self.sig_batch, generator=self.gen))
        z = np.concatenate(z, axis=0)
        
        # split data
        x = z[:, :-2]
        m = np.reshape(np.ones_like(z[:, -2]) * self.mass, newshape=(-1, 1))
        y = np.reshape(z[:, -1], newshape=(-1, 1))

        return dict(x=x, m=m), y

    @classmethod
    def get_data(cls, dataset: Dataset, features: list, mass: float, case: int, train_batch=128, eval_batch=1024, **kwargs):
        # split data
        train, valid, test = train_val_test_split(dataset, **kwargs)
        
        # create sequences
        train_seq = cls(signal=train[0], background=train[1], mass=mass, case=case, batch_size=train_batch,
                        features=features, balance=True)

        valid_seq = cls(signal=valid[0], background=valid[1], mass=mass, case=case, batch_size=eval_batch,
                        features=features, balance=False)

        test_seq = cls(signal=test[0], background=test[1], mass=mass, case=case, batch_size=eval_batch, 
                       features=features, balance=False)
        
        # create tf.Datasets
        train_ds = utils.dataset_from_sequence(train_seq)
        valid_ds = utils.dataset_from_sequence(valid_seq)
        test_ds = utils.dataset_from_sequence(test_seq)
        
        return train_ds, valid_ds, test_ds

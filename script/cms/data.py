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


# def get_data_baseline(dataset: Dataset, features: list, train_batch=128, eval_batch=1024, **kwargs):
#     # split data
#     train, valid, test = train_val_test_split(dataset, **kwargs)
    
#     # create sequences
#     train_seq = BaselineSequence(signal=train[0], background=train[1], batch_size=train_batch, 
#                                  features=features)

#     valid_seq = BaselineSequence(signal=valid[0], background=valid[1], batch_size=eval_batch,
#                                  features=features, balance_signal=False)

#     test_seq = BaselineSequence(signal=test[0], background=test[1], batch_size=eval_batch, 
#                                 features=features, balance_signal=False)
    
#     # create tf.Datasets
#     train_ds = utils.dataset_from_sequence(train_seq)
#     valid_ds = utils.dataset_from_sequence(valid_seq)
#     test_ds = utils.dataset_from_sequence(test_seq)
    
#     return train_ds, valid_ds, test_ds


class BalancedSequence(tf.keras.utils.Sequence):
    """keras.Sequence that balances the signal (each mA has the same number of events), and backgrounds"""

    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, delta=50, balance_signal=True, balance_bkg=True, seed=utils.SEED,
                 sample_mass=False):
        # select data
        self.sig = signal[features + ['mA', 'type', 'dimuon_M']]
        self.names = background['name']
        self.bkg = background[features + ['mA', 'type', 'dimuon_M']]
        
        self.mass = np.sort(self.sig['mA'].unique())
        self.mass_intervals = np.array(Dataset.MASS_INTERVALS)
        self.mass_intervals = self.mass_intervals[:, 1]
        
        self.should_sample_mass = bool(sample_mass)
        
        # self.rnd = np.random.RandomState(seed)  # "slow" but enough for pd.DataFrame.sample
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
            # take mass from corresponding mass interval (m - delta, m + delta)
            mask = y == 0.0
            dimuon_mass = z[:, -1]
            
            idx = np.digitize(dimuon_mass[mask], self.mass_intervals, right=True)
            idx = np.clip(idx, a_min=0, a_max=len(self.mass) - 1)
        
            m[mask] = self.mass[idx]
    
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        return dict(x=x, m=m), y

    @classmethod
    def get_data(cls, dataset: Dataset, features: list, case: int, train_batch=128, eval_batch=1024, **kwargs):
        # split data
        train, valid, test = train_val_test_split(dataset, **kwargs)
        
        # create sequences
        train_seq = cls(signal=train[0], background=train[1], batch_size=train_batch, 
                        features=features, sample_mass=case == 1)

        valid_seq = cls(signal=valid[0], background=valid[1], batch_size=eval_batch,
                        features=features, balance_signal=False, balance_bkg=False, sample_mass=False)

        test_seq = cls(signal=test[0], background=test[1], batch_size=eval_batch, 
                       features=features, balance_signal=False, balance_bkg=False, sample_mass=False)
        
        # create tf.Datasets
        train_ds = utils.dataset_from_sequence(train_seq)
        valid_ds = utils.dataset_from_sequence(valid_seq)
        test_ds = utils.dataset_from_sequence(test_seq)
        
        return train_ds, valid_ds, test_ds


# def get_data_balanced(dataset: Dataset, features: list, case: int, train_batch=128, eval_batch=1024, **kwargs):
#     # split data
#     train, valid, test = train_val_test_split(dataset, **kwargs)
    
#     # create sequences
#     train_seq = BalancedSequence(signal=train[0], background=train[1], batch_size=train_batch, 
#                                  features=features, sample_mass=case == 1)

#     valid_seq = BalancedSequence(signal=valid[0], background=valid[1], batch_size=eval_batch,
#                                  features=features, balance_signal=False, balance_bkg=False, sample_mass=False)

#     test_seq = BalancedSequence(signal=test[0], background=test[1], batch_size=eval_batch, 
#                                 features=features, balance_signal=False, balance_bkg=False, sample_mass=False)
    
#     # create tf.Datasets
#     train_ds = utils.dataset_from_sequence(train_seq)
#     valid_ds = utils.dataset_from_sequence(valid_seq)
#     test_ds = utils.dataset_from_sequence(test_seq)
    
#     return train_ds, valid_ds, test_ds


class MassBalancedSequence(tf.keras.utils.Sequence):
    """tf.keras.Sequence with balanced batches for each mass"""
    
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, features: list,
                 beta=0.5, delta=50, balance=True, weight=False, signal_weight=None, sample_mass=False,
                 seed=utils.SEED):
        columns = features + ['mA', 'type', 'dimuon_M']
        self.indices = {'features': -4, 'mass': -4, 'label': -3, 'dimuon_M': -2, 'weight': -1}
        
        sig = signal
        bkg = background
        w_max = bkg['weight'].max()
        
        self.mass = np.sort(sig['mA'].unique())
        self.mass_intervals = np.array(Dataset.MASS_INTERVALS)
        self.mass_intervals = self.mass_intervals[:, 1]
        
        self.should_balance = bool(balance)
        self.should_weight = bool(weight)
        self.should_sample_mass = bool(sample_mass)
        
        self.gen = utils.get_random_generator(seed)
        self.num_batches = (sig.shape[0] + bkg.shape[0]) // batch_size
        
        if self.should_balance:
            self.batch_sig = (batch_size / 2) // self.mass.shape[0]
            self.batch_bkg = ((batch_size / 2) / self.mass.shape[0]) // len(bkg['name'].unique()) 
            
            self.signals = {}
            self.bkgs = {}
            
            s_ = {m: sig[sig['mA'] == m] for m in self.mass}
            b_ = {m: bkg[(bkg['dimuon_M'] > m - delta) & (bkg['dimuon_M'] < m + delta)] for m in self.mass}
            
            # num. sig and bkg such that #sig ~ #bkg
            num_s = list(map(lambda x: x.shape[0], s_.values()))
            num_b = list(map(lambda x: x.shape[0], b_.values()))
            
            eq_idx = np.argmin(np.abs(np.array(num_s) - num_b))
            s_eq = num_s[eq_idx]
            b_eq = num_b[eq_idx]
            
            if s_eq > b_eq:
                ws_eq = b_eq / s_eq
                wb_eq = 1.0
            else:
                ws_eq = 1.0
                wb_eq = s_eq / b_eq
            
            # compute weights
            ws = []
            wb = []
            
            for i, m in enumerate(self.mass):
                 # weight signal such that the summed weighted signal equals `s_eq`
                w_s = ws_eq * (s_eq / num_s[i])
                w_b = wb_eq * (b_eq / num_b[i])
                
                ws.append(np.power(w_s, beta) if w_s > 1.0 else w_s)
                wb.append(np.power(w_b, beta) if w_b > 1.0 else w_b)
            
            # slice data
            for i, m in enumerate(self.mass):
                s = s_[m]
                b = b_[m]
                
                w_s = ws[i]
                w_b = wb[i]
                
                # signal
                if signal_weight is None:
                    w_s = ws[i]
                else:
                    w_s = float(signal_weight)
                    
                w_s = np.ones((num_s[i], 1)) * w_s
                self.signals[m] = np.concatenate([s[columns].values, w_s], axis=-1)
                
                # background
                self.bkgs[m] = {}
                
                for name in b['name'].unique():
                    b_name = b[b['name'] == name]
                    w_name = np.ones((b_name['weight'].shape[0], 1)) * w_b
                    
                    self.bkgs[m][name] = np.concatenate([b_name[columns].values, w_name], axis=-1)
        else:
            s = sig[columns + ['weight']]
            b = bkg[columns + ['weight']]
            
            self.all = np.concatenate([s.values, b.values], axis=0)
            self.batch_size = int(batch_size)
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        if self.should_balance:
            # sample signal (each mA)
            z = [np_sample(s, amount=self.batch_sig, generator=self.gen) for s in self.signals.values()]
            
            # sample background (each mA and process)
            for bkg in self.bkgs.values():
                for b in bkg.values():
                    z.append(np_sample(b, amount=self.batch_bkg, generator=self.gen))
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
            dimuon_mass = z[:, self.indices['dimuon_M']]
            
            idx = np.digitize(dimuon_mass[mask], self.mass_intervals, right=True)
            idx = np.clip(idx, a_min=0, a_max=len(self.mass) - 1)
        
            m[mask] = self.mass[idx]
    
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        if self.should_weight:
            w = z[:, self.indices['weight']]
            w = np.reshape(w, newshape=(-1, 1))
            
            return dict(x=x, m=m), y, w
        
        return dict(x=x, m=m), y

    @classmethod
    def get_data(cls, dataset: Dataset, features: list, case: int, train_batch=128, eval_batch=1024, 
                 weight=False, **kwargs):
        # split data
        train, valid, test = train_val_test_split(dataset)
        
        # create sequences
        train_seq = cls(signal=train[0], background=train[1], batch_size=train_batch, weight=weight,
                        features=features, sample_mass=case == 1, balance=True, **kwargs)

        valid_seq = cls(signal=valid[0], background=valid[1], batch_size=eval_batch, weight=False,
                        features=features, sample_mass=False, balance=False, **kwargs)

        test_seq = cls(signal=test[0], background=test[1], batch_size=eval_batch, weight=False,
                       features=features, sample_mass=False, balance=False, **kwargs)
        
        # create tf.Datasets
        train_ds = utils.dataset_from_sequence(train_seq, sample_weights=bool(weight))
        valid_ds = utils.dataset_from_sequence(valid_seq)
        test_ds = utils.dataset_from_sequence(test_seq)
        
        return train_ds, valid_ds, test_ds


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

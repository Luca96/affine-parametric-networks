"""CMS/data"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

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


def train_valid_split(dataset: Dataset, valid_size=0.25, seed=utils.SEED):
    sig = dataset.signal
    bkg = dataset.background
    
    # validation split
    train_sig, valid_sig = train_test_split(sig, test_size=valid_size, random_state=seed)
    train_bkg, valid_bkg = train_test_split(bkg, test_size=valid_size, random_state=seed)
    
    return (train_sig, train_bkg), (valid_sig, valid_bkg)


class SimpleEvalSequence(tf.keras.utils.Sequence):
    """Does 1-vs-all validation: each time all (weighted) background is used for a mA"""
    
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, weight_column='weight', seed=utils.SEED, sample_mass=True, 
                 normalize_signal_weights=False, sample_mass_initially=False, 
                 replicate_bkg=False, shuffle_on_epoch=False, **kwargs):
        assert batch_size >= 1
        
        columns = features + ['mA', 'type']
        self.indices = {'features': -2, 'mass': -2, 'label': -1}
        
        if isinstance(weight_column, str):
            assert (weight_column in signal) and (weight_column in background)
            self.should_weight = True

            columns += [weight_column]

            self.indices = {k: idx - 1 for k, idx in self.indices.items()}  # decrease index by "-1"
            self.indices['weight'] = -1  # add new index for sample-weights
        else:
            self.should_weight = False

        self.gen = utils.get_random_generator(seed)
        self.batch_size = int(batch_size)
        self.should_sample_mass = bool(sample_mass)
        self.should_shuffle_on_epoch = bool(shuffle_on_epoch)

        self.mass = np.sort(signal['mA'].unique())
        self.data = np.concatenate([signal[columns].values, 
                                    background[columns].values], axis=0)
        
        if replicate_bkg and self.should_sample_mass:
            self.data = np.concatenate([self.data] + [background[columns].values] * (len(self.mass) - 1),
                                       axis=0)
        
        self.gen.shuffle(self.data)

        # sample some datapoints if last batch is not full
        remaining_samples = len(self.data) % self.batch_size

        if remaining_samples > 0:
            samples = self.gen.choice(self.data, size=remaining_samples)
            self.data = np.concatenate([self.data, samples], axis=0)
        
        if bool(normalize_signal_weights):
            self._normalize_signal_weights()
            
        if bool(sample_mass_initially):
            mask = self.data[:, self.indices['label']] == 0.0
            self.data[mask][:, self.indices['mass']] = self.gen.uniform(low=self.mass.min(), 
                                                                        high=self.mass.max(), size=np.sum(mask))
    
    def __len__(self):
        return self.data.shape[0] // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        stop_idx = start_idx + self.batch_size

        z = self.data[start_idx:stop_idx]

        # split data (features, mass, label, dimuon_mass, weight)
        x = z[:, :self.indices['features']]
        m = z[:, self.indices['mass']].reshape(-1, 1)
        y = z[:, self.indices['label']].reshape(-1, 1)
        
        # sample mA for background
        if self.should_sample_mass:
            mask = y == 0.0
            m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True)
        
        if self.should_weight:
            w = z[:, self.indices['weight']]
            return dict(x=x, m=m), y, w.reshape(-1, 1)
        
        return dict(x=x, m=m), y

    def on_epoch_end(self):
        if self.should_shuffle_on_epoch:
            self.gen.shuffle(self.data)
            utils.free_mem()

    def to_tf_dataset(self, **kwargs):
        return utils.dataset_from_sequence(self, sample_weights=self.should_weight, **kwargs)
    
    def _normalize_signal_weights(self):
        if self.should_weight:
            s_mask = self.data[:, self.indices['label']] == 1.0
            b_mask = self.data[:, self.indices['label']] == 0.0

            w_b = self.data[b_mask][:, self.indices['weight']]
            w_s = np.sum(w_b) / np.sum(s_mask)

            self.data[s_mask][:, self.indices['weight']] = w_s

            
class EvalSequence(SimpleEvalSequence):
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, intervals: list, 
                 batch_size: int, features: list, weight_column=None, seed=utils.SEED, 
                 normalize_signal_weights=False, **kwargs):
        assert isinstance(intervals, (list, np.ndarray))
        assert batch_size >= 1

        columns = features + ['mA', 'type']
        self.indices = {'features': -2, 'mass': -2, 'label': -1}
        
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

        self.gen = utils.get_random_generator(seed)
        self.batch_size = int(batch_size)

        self.mass = np.sort(sig['mA'].unique())
        assert len(self.mass) == len(intervals)

        # select data
        self.data = []
        count = 0

        for mass, (low, up) in zip(self.mass, intervals):
            s = sig[sig['mA'] == mass][columns]
            b = bkg[(bkg['dimuon_mass'] > low) & (bkg['dimuon_mass'] < up)][columns]

            self.data.append(s.values)

            # set mA for corresponding background (in interval)
            b_values = b.values
            b_values[:, self.indices['mass']] = mass

            self.data.append(b_values)

            count += s.shape[0] + b.shape[0]

        self.data = np.concatenate(self.data, axis=0)
        self.gen.shuffle(self.data)

        # sample some datapoints if last batch is not full
        remaining_samples = count % self.batch_size

        if remaining_samples > 0:
            samples = self.gen.choice(self.data, size=remaining_samples)
            self.data = np.concatenate([self.data, samples], axis=0)
        
        if bool(normalize_signal_weights):
            self._normalize_signal_weights()


class AbstractSequence(tf.keras.utils.Sequence):
    """Base class for custom tf.keras.Sequence"""
    
    def __init__(self, *args, seed=utils.SEED, **kwargs):
        self.should_weight = False
        
        self.seed = seed
        self.gen = utils.get_random_generator(seed)
    
    def to_tf_dataset(self):
        return utils.dataset_from_sequence(self, sample_weights=self.should_weight)
    
    def _shuffle_data(self):
        self.gen.shuffle(self.data)
        utils.free_mem()

    @classmethod
    def get_data(cls, dataset, train_batch=1024, eval_batch=1024, eval_cls=SimpleEvalSequence,
                 **kwargs):
        # split data
        train, valid = train_valid_split(dataset, seed=kwargs.get('seed', utils.SEED))
        
        # create sequences
        features = kwargs.pop('features', dataset.columns['feature'])
        
        train_seq = cls(signal=train[0], background=train[1], batch_size=int(train_batch),
                        features=features, **kwargs)
        
        valid_seq = eval_cls(signal=valid[0], background=valid[1], features=features,
                             batch_size=int(eval_batch), **kwargs)
        # return tf.Datasets
        return [seq.to_tf_dataset() for seq in [train_seq, valid_seq]]


# TODO: rename to `IdenticalSequence` or `FixedSequence`?
class OneVsAllSequence(AbstractSequence):
    """Implements the '1-vs-all' mA sampling strategy for 'same' distribution"""
    
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, **kwargs):
        super().__init__(**kwargs)
        
        s = signal[features + ['mA', 'type']]
        b = background[features + ['mA', 'type']]
        
        self.mass = np.sort(s['mA'].unique())
        self.batch_size = int(batch_size)
        
        self.data = np.concatenate([s.values, b.values], axis=0)
        self.gen.shuffle(self.data)
        
    def __len__(self):
        return len(self.data) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        stop_idx = start_idx + self.batch_size

        z = self.data[start_idx:stop_idx]

        # split data (features, mA, label)
        x = z[:, :-2]
        m = z[:, -2].reshape(-1, 1)
        y = z[:, -1].reshape(-1, 1)
        
        # sample mA for background (1-vs-all)
        mask = y == 0.0
        m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True)
        
        return dict(x=x, m=m), y

    def on_epoch_end(self):
        self._shuffle_data()

    
class IntervalSequence(EvalSequence, AbstractSequence):
    """Implements mA assignment based on given `intervals`"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.should_sample_mass = False

    def on_epoch_end(self):
        self._shuffle_data()


class BalancedSequence(AbstractSequence):
    """keras.Sequence that balances the signal (each mA has the same number of events), and backgrounds"""

    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, balance_signal=True, balance_bkg=True, seed=utils.SEED,
                 sample_mass=False, bins: list = None):
        super().__init__()
        
        # select data
        self.sig = signal[features + ['mA', 'type', 'dimuon_mass']]
        self.names = background['name']
        self.bkg = background[features + ['mA', 'type', 'dimuon_mass']]
        
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

        self.gen = utils.get_random_generator(seed)
        self.half_batch = batch_size // 2
        
        self.should_sample_mass = bool(sample_mass)
        self.should_balance_sig = bool(balance_signal)
        self.should_balance_bkg = bool(balance_bkg)
        
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


class MassBalancedSequence(AbstractSequence):
    """tf.keras.Sequence with balanced batches for each mass"""
    
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, delta=50, balance=True, weight_column=None, sample_mass=False,
                 intervals: list = None, **kwargs):
        super().__init__(**kwargs)
        
        columns = features + ['mA', 'type', 'dimuon_mass']
        self.indices = {'features': -3, 'mass': -3, 'label': -2, 'dimuon_mass': -1}
        
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
                deltas = [(delta, delta)] * len(self.mass)

            elif isinstance(delta, tuple):
                deltas = [delta] * len(self.mass)
            else:
                assert isinstance(delta, (list, np.ndarray))
                assert len(delta) == len(self.mass)

                deltas = delta

            deltas = np.array(deltas)
            
            self.mass_intervals = np.stack([self.mass - deltas[:, 0],
                                            self.mass + deltas[:, 1]], axis=-1)
        else:
            self.mass_intervals = np.array(intervals)

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
        self.mass_intervals = self.mass_intervals.astype(np.float32)

        self.mass_intervals[0][0] = min(sig['dimuon_mass'].min(), bkg['dimuon_mass'].min()) - 1.0
        self.mass_intervals[-1][1] = max(sig['dimuon_mass'].max(), bkg['dimuon_mass'].max()) + 1.0

        self.should_balance = bool(balance)
        self.should_sample_mass = bool(sample_mass)
        
        self.num_batches = (sig.shape[0] + bkg.shape[0]) // batch_size
        
        if self.should_balance:
            self.batch_sig = (batch_size / 2) // self.mass.shape[0]
            self.batch_bkg = ((batch_size / 2) / self.mass.shape[0]) // len(bkg['name'].unique()) 
            
            self.signals = {}
            self.bkgs = {}
            
            s_ = {m: sig[sig['mA'] == m] for m in self.mass}
            b_ = {m: bkg[(bkg['dimuon_mass'] > low) & (bkg['dimuon_mass'] < up)] for m, (low, up) in zip(self.mass, self.mass_intervals)}
        
            # slice data
            for i, m in enumerate(self.mass):
                s = s_[m]
                b = b_[m]
                
                # signal
                self.signals[m] = s[columns].values
                
                # background
                self.bkgs[m] = {}
                
                for name in b['name'].unique():
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
        
        # split data (features, mass, label, dimuon_mass, weight)
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

            if self.should_balance:
                m[mask] = m_bkg
            else:
                dimuon_mass = z[:, self.indices['dimuon_mass']]

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
    
    
class FullBalancedSequence(MassBalancedSequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not self.should_sample_mass:
            # fixed mA distribution: so change bkg's mA with a value sampled from signal's mA
            for m, bkg in self.bkgs.items():
                for b in bkg.values():
                    b[:, self.indices['mass']] = self.gen.choice(self.mass, size=len(b))


class UniformSequence(AbstractSequence):
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, **kwargs):
        assert batch_size >= 1
        super().__init__(**kwargs)
        
        columns = features + ['mA', 'type']

        self.indices = {'features': -2, 'mass': -2, 'label': -1}
        self.batch_size = int(batch_size)

        self.mass = np.sort(signal['mA'].unique())
        self.mass_interval = (self.mass.min(), self.mass.max())

        # select data
        self.data = np.concatenate([signal[columns].values, 
                                    background[columns].values], axis=0)
        self.gen.shuffle(self.data)

        # sample some datapoints if last batch is not full
        remaining_samples = len(self.data) % self.batch_size

        if remaining_samples > 0:
            samples = self.gen.choice(self.data, size=remaining_samples)
            self.data = np.concatenate([self.data, samples], axis=0)

    def __len__(self):
        return self.data.shape[0] // self.batch_size
    
    def _uniform_sample(self, amount: int):
        return self.gen.uniform(low=self.mass_interval[0], high=self.mass_interval[1], size=int(amount))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        stop_idx = start_idx + self.batch_size

        z = self.data[start_idx:stop_idx]

        # split data (features, mass, label, dimuon_mass, weight)
        x = z[:, :self.indices['features']]
        m = z[:, self.indices['mass']].reshape(-1, 1)
        y = z[:, self.indices['label']].reshape(-1, 1)
        
        # sample mA for background
        mask = y == 0.0
        m[mask] = self._uniform_sample(amount=np.sum(mask))
        
        return dict(x=x, m=m), y

    def on_epoch_end(self):
        self._shuffle_data()


class BalancedUniformSequence(AbstractSequence):
    """tf.keras.Sequence with balanced batches for each mass"""
    
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, balance_signal=True, balance_bkg=True, sample_mass=False,
                 initial_uniform_mass=False, **kwargs):
        super().__init__(**kwargs)
        
        columns = features + ['mA', 'type']
        self.indices = {'features': -2, 'mass': -2, 'label': -1}
        
        s = signal
        b = background
        
        self.mass = np.sort(signal['mA'].unique())
        self.mass_interval = (self.mass.min(), self.mass.max())

        self.should_sample_mass = bool(sample_mass)
        self.should_balance_signal = bool(balance_signal)
        self.should_balance_background = bool(balance_bkg)
        
        # signal mA balancing
        if self.should_balance_signal: 
            self.sig_batch = (batch_size / 2) // len(self.mass)
            self.signals = {m: s[s['mA'] == m][columns].values for m in self.mass}
            
            actual_batch = len(self.mass) * self.sig_batch
        else:
            actual_batch = batch_size // 2
            
            self.sig_batch = batch_size // 2
            self.signals = {0: signal[columns].values}  # all signal ("0" is a placeholder for key)
        
        # background-process balancing
        if self.should_balance_background:
            processes = b['name'].unique()
            
            self.bkg_batch = (batch_size / 2) // len(processes)
            self.bkgs = {name: b[b['name'] == name][columns].values for name in processes}
            
            actual_batch += len(processes) * self.bkg_batch
        else:
            actual_batch += batch_size // 2
            
            self.bkg_batch = batch_size // 2
            self.bkgs = {0: background[columns].values}  # all background
        
        self.num_batches = int((s.shape[0] + b.shape[0]) / actual_batch)
        
        if bool(initial_uniform_mass):
            for m, bkg in self.bkgs.items():
                bkg[:, self.indices['mass']] = self._uniform_sample(amount=len(bkg))

    def __len__(self):
        return self.num_batches
    
    def _uniform_sample(self, amount: int):
        return self.gen.uniform(low=self.mass_interval[0], high=self.mass_interval[1], size=int(amount))
    
    def __getitem__(self, idx):
        # balanced sampling
        z = [np_sample(s, amount=self.sig_batch, generator=self.gen) for s in self.signals.values()]
        z += [np_sample(b, amount=self.bkg_batch, generator=self.gen) for b in self.bkgs.values()]
        
        z = np.concatenate(z, axis=0)
        
        # split data (features, mass, label)
        x = z[:, :self.indices['features']]
        m = z[:, self.indices['mass']]
        y = z[:, self.indices['label']]
        
        if self.should_sample_mass:
            # sample mass uniformly within mA range
            mask = y == 0.0
            m[mask] = self._uniform_sample(amount=np.sum(mask))
    
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        return dict(x=x, m=m), y

    def on_epoch_end(self):
        pass


class BalancedIdenticalSequence(BalancedUniformSequence):
    """tf.keras.Sequence with balanced batches for each mass; identical mA distribution"""
    
    def __init__(self, *args, **kwargs):
        kwargs.pop('initial_uniform_mass', None)
        super().__init__(*args, initial_uniform_mass=False, **kwargs)
            
    def __getitem__(self, idx):
        # balanced sampling
        z = [np_sample(s, amount=self.sig_batch, generator=self.gen) for s in self.signals.values()]
        z += [np_sample(b, amount=self.bkg_batch, generator=self.gen) for b in self.bkgs.values()]
        
        z = np.concatenate(z, axis=0)
        
        # split data (features, mass, label)
        x = z[:, :self.indices['features']]
        m = z[:, self.indices['mass']]
        y = z[:, self.indices['label']]
        
        if self.should_sample_mass:
            # sample mass "identically" within mA range
            mask = y == 0.0
            m[mask] = self.gen.choice(self.mass, size=np.sum(mask))
    
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        return dict(x=x, m=m), y

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
    """Does 1-vs-all validation: each time all (weighted) background is used for a mass"""
    
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, weight_column='weight', seed=utils.SEED, sample_mass=True, 
                 normalize_signal_weights=False, sample_mass_initially=False, 
                 replicate_bkg=False, shuffle_on_epoch=False, **kwargs):
        assert batch_size >= 1
        
        columns = features + ['mass', 'type']
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

        self.mass = np.sort(signal['mass'].unique())
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

        # split data (features, mass, label, weight)
        x = z[:, :self.indices['features']]
        m = z[:, self.indices['mass']].reshape(-1, 1)
        y = z[:, self.indices['label']].reshape(-1, 1)
        
        # sample mass for background
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


class IdenticalSequence(AbstractSequence):
    """Implements the identical mass sampling strategy for mass feature assignment"""
    
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, **kwargs):
        super().__init__(**kwargs)
        
        s = signal[features + ['mass', 'type']]
        b = background[features + ['mass', 'type']]
        
        self.mass = np.sort(s['mass'].unique())
        self.batch_size = int(batch_size)
        
        self.data = np.concatenate([s.values, b.values], axis=0)
        self.gen.shuffle(self.data)
        
    def __len__(self):
        return len(self.data) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        stop_idx = start_idx + self.batch_size

        z = self.data[start_idx:stop_idx]

        # split data (features, mass, label)
        x = z[:, :-2]
        m = z[:, -2].reshape(-1, 1)
        y = z[:, -1].reshape(-1, 1)
        
        # sample mass for background (identical strategy)
        mask = y == 0.0
        m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True)
        
        return dict(x=x, m=m), y

    def on_epoch_end(self):
        self._shuffle_data()

    
class UniformSequence(AbstractSequence):
    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
                 features: list, **kwargs):
        assert batch_size >= 1
        super().__init__(**kwargs)
        
        columns = features + ['mass', 'type']

        self.indices = {'features': -2, 'mass': -2, 'label': -1}
        self.batch_size = int(batch_size)

        self.mass = np.sort(signal['mass'].unique())
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
        
        # sample mass for background
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
        
        columns = features + ['mass', 'type']
        self.indices = {'features': -2, 'mass': -2, 'label': -1}
        
        s = signal
        b = background
        
        self.mass = np.sort(signal['mass'].unique())
        self.mass_interval = (self.mass.min(), self.mass.max())

        self.should_sample_mass = bool(sample_mass)
        self.should_balance_signal = bool(balance_signal)
        self.should_balance_background = bool(balance_bkg)
        
        # signal mass balancing
        if self.should_balance_signal: 
            self.sig_batch = (batch_size / 2) // len(self.mass)
            self.signals = {m: s[s['mass'] == m][columns].values for m in self.mass}
            
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
            # sample mass uniformly within mass range
            mask = y == 0.0
            m[mask] = self._uniform_sample(amount=np.sum(mask))
    
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        return dict(x=x, m=m), y

    def on_epoch_end(self):
        pass


class BalancedIdenticalSequence(BalancedUniformSequence):
    """tf.keras.Sequence with balanced batches for each mass; identical mass distribution"""
    
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
            # sample mass "identically" within mass range
            mask = y == 0.0
            m[mask] = self.gen.choice(self.mass, size=np.sum(mask))
    
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        return dict(x=x, m=m), y

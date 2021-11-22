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


# class BaselineSequence(tf.keras.utils.Sequence):
#     def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
#                  features: list, delta=50, balance_signal=True, seed=utils.SEED):
#         self.sig = signal
#         self.bkg = background
        
#         self.mass = sorted(self.sig['mA'].unique())
#         self.signals = {m: self.sig[self.sig['mA'] == m] for m in self.mass}
        
#         self.rnd = np.random.RandomState(seed)  # "slow" but enough for pd.DataFrame.sample
#         self.gen = utils.get_random_generator(seed)   # "fast" random generator for `np.random.choice`
        
#         self.should_balance = bool(balance_signal)
        
#         self.features = features
#         self.bkg_batch = batch_size // 2
        
#         if self.should_balance:
#             self.sig_batch = (batch_size / 2) // len(self.signals.keys())
#         else:
#             self.sig_batch = batch_size // 2
    
#     def __len__(self):
#         return self.sig.shape[0] // self.bkg_batch  # bkg_batch = half batch-size
    
#     def __getitem__(self, idx):
#         if self.should_balance:
#             df = [sample(sig, amount=self.sig_batch, seed=self.rnd) for sig in self.signals.values()]
#         else:
#             df = [sample(self.sig, amount=self.sig_batch, seed=self.rnd)]
        
#         df.append(sample(self.bkg, amount=self.bkg_batch, seed=self.rnd))
        
#         df = pd.concat(df, axis=0)
        
#         x = df[self.features].values
#         m = np.reshape(df['mA'].values, newshape=(-1, 1))
#         y = np.reshape(df['type'].values, newshape=(-1, 1))
        
#         # sample mass (from signal's mA) for background events
#         mask = np.squeeze(y == 0.0)
#         m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True).reshape((-1, 1))
        
#         return dict(x=x, m=m), y


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


def get_data_baseline(dataset: Dataset, features: list, train_batch=128, eval_batch=1024, **kwargs):
    # split data
    train, valid, test = train_val_test_split(dataset, **kwargs)
    
    # create sequences
    train_seq = BaselineSequence(signal=train[0], background=train[1], batch_size=train_batch, 
                                 features=features)

    valid_seq = BaselineSequence(signal=valid[0], background=valid[1], batch_size=eval_batch,
                                 features=features, balance_signal=False)

    test_seq = BaselineSequence(signal=test[0], background=test[1], batch_size=eval_batch, 
                                features=features, balance_signal=False)
    
    # create tf.Datasets
    train_ds = utils.dataset_from_sequence(train_seq)
    valid_ds = utils.dataset_from_sequence(valid_seq)
    test_ds = utils.dataset_from_sequence(test_seq)
    
    return train_ds, valid_ds, test_ds


# class BalancedSequence(tf.keras.utils.Sequence):
#     """keras.Sequence that balances the signal (each mA has the same number of events), and backgrounds"""

#     def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, batch_size: int, 
#                  features: list, delta=50, balance_signal=True, balance_bkg=True, seed=utils.SEED,
#                  sample_mass=False):
#         self.sig = signal
#         self.bkg = background
        
#         self.mass = np.sort(self.sig['mA'].unique())
#         self.mass_intervals = np.array(Dataset.MASS_INTERVALS)
#         self.mass_intervals = self.mass_intervals[:, 1]
        
#         self.should_sample_mass = bool(sample_mass)
        
#         self.rnd = np.random.RandomState(seed)  # "slow" but enough for pd.DataFrame.sample
#         self.gen = utils.get_random_generator(seed)   # "fast" random generator for `np.random.choice`
        
#         self.should_balance_sig = bool(balance_signal)
#         self.should_balance_bkg = bool(balance_bkg)
        
#         self.features = features
#         self.half_batch = batch_size // 2
        
#         if self.should_balance_sig:
#             self.signals = {m: self.sig[self.sig['mA'] == m] for m in self.mass}
            
#             self.sig_batch = self.half_batch // len(self.signals.keys())
#         else:
#             self.sig_batch = self.half_batch
            
#         if self.should_balance_bkg:
#             self.bkgs = {k: self.bkg[self.bkg['name'] == k] for k in self.bkg['name'].unique()}
            
#             self.bkg_batch = self.half_batch // len(self.bkgs.keys())
#         else:
#             self.bkg_batch = self.half_batch
    
#     def __len__(self):
# #         return (self.sig.shape[0] // self.half_batch) + (self.bkg.shape[0] // self.half_batch)
#         return self.sig.shape[0] // self.half_batch
    
#     def __getitem__(self, idx):
#         if self.should_balance_sig:
#             df = [sample(sig, amount=self.sig_batch, seed=self.rnd) for sig in self.signals.values()]
#         else:
#             df = [sample(self.sig, amount=self.half_batch, seed=self.rnd)]
        
#         if self.should_balance_bkg:
#             df.extend([sample(bkg, amount=self.bkg_batch, seed=self.rnd) for bkg in self.bkgs.values()])
#         else:
#             df.append(sample(self.bkg, amount=self.half_batch, seed=self.rnd))
        
#         df = pd.concat(df, axis=0)
        
#         x = df[self.features].values
#         m = np.reshape(df['mA'].values, newshape=(-1, 1))
#         y = np.reshape(df['type'].values, newshape=(-1, 1))
        
#         if self.should_sample_mass:
#             # sample mass (from signal's mA) for background events -> uses all background events
#             mask = np.squeeze(y == 0.0)
#             m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True).reshape((-1, 1))
#         else:
#             # take mass from corresponding mass interval (m - delta, m + delta)
#             mask = np.squeeze(y == 0.0)
            
#             idx = np.digitize(df['dimuon_M'].values[mask], self.mass_intervals, right=True)
#             idx = np.clip(idx, a_min=0, a_max=len(self.mass) - 1)
        
#             m[mask] = self.mass[idx].reshape((-1, 1))
        
#         return dict(x=x, m=m), y


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
        # m = np.reshape(z[:, -2], newshape=(-1, 1))
        # y = np.reshape(z[:, -1], newshape=(-1, 1))
        
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


def get_data_balanced(dataset: Dataset, features: list, case: int, train_batch=128, eval_batch=1024, **kwargs):
    # split data
    train, valid, test = train_val_test_split(dataset, **kwargs)
    
    # create sequences
    train_seq = BalancedSequence(signal=train[0], background=train[1], batch_size=train_batch, 
                                 features=features, sample_mass=case == 1)

    valid_seq = BalancedSequence(signal=valid[0], background=valid[1], batch_size=eval_batch,
                                 features=features, balance_signal=False, balance_bkg=False, sample_mass=False)

    test_seq = BalancedSequence(signal=test[0], background=test[1], batch_size=eval_batch, 
                                features=features, balance_signal=False, balance_bkg=False, sample_mass=False)
    
    # create tf.Datasets
    train_ds = utils.dataset_from_sequence(train_seq)
    valid_ds = utils.dataset_from_sequence(valid_seq)
    test_ds = utils.dataset_from_sequence(test_seq)
    
    return train_ds, valid_ds, test_ds

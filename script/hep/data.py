"""HEPMASS/data"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from sklearn.model_selection import train_test_split

from script import utils

from typing import Union


def np_sample(x: np.ndarray, amount: int, generator):
    """Samples a random `amount` of events from given np.ndarray `x`"""
    amount = int(amount)

    indices = generator.choice(np.arange(x.shape[0]), size=amount, replace=True)
    return x[indices]


def train_valid_split(dataset, valid_size=0.25, seed=utils.SEED):
    sig = dataset.signal
    bkg = dataset.background
    
    # validation split
    train_sig, valid_sig = train_test_split(sig, test_size=valid_size, random_state=seed)
    train_bkg, valid_bkg = train_test_split(bkg, test_size=valid_size, random_state=seed)
    
    return (train_sig, train_bkg), (valid_sig, valid_bkg)


class EvalSequence(tf.keras.utils.Sequence):
    def __init__(self, data, signal: pd.DataFrame, bkg: pd.DataFrame, batch_size: int, 
                 features: list, mass: list = None, all_bkg=False, seed=utils.SEED, **kwargs):
        assert batch_size >= 1
        
        mA = data.columns['mA']
        columns = features + [mA, data.columns['label']]
        self.indices = {'features': -2, 'mass': -2, 'label': -1}
        
        self.gen = utils.get_random_generator(seed)
        self.batch_size = int(batch_size)

        # mass
        if isinstance(mass, (list, np.ndarray)):
            self.mass = np.sort(mass)
            should_take_all_bkg = False or all_bkg
        else:
            self.mass = data.unique_signal_mass
            should_take_all_bkg = True
        
        # background selection
        if should_take_all_bkg:
            b = bkg[columns]  # pick all background
            w_b = 1.0 / len(data.unique_signal_mass)
        else:
            b = pd.concat([bkg[bkg[mA] == m][columns] for m in self.mass])
            w_b = 1.0 / len(self.mass)
            
        # select data
        count = 0
        weights = []  # sample-weights
        self.data = []
        
        for mass in self.mass:
            s = signal[signal[mA] == mass][columns]
            
            self.data.append(s.values)
            weights.append(np.ones((s.shape[0], 1), dtype=np.float32))  # signal-weights

            # set mA for background
            b_values = b.values
            b_values[:, self.indices['mass']] = mass

            self.data.append(b_values)
            weights.append(np.ones((b.shape[0], 1), dtype=np.float32) * w_b)  # bkg-weights
            
            count += s.shape[0] + b.shape[0]

        self.data = np.concatenate(self.data, axis=0)
        self.gen.shuffle(self.data)

        # sample some datapoints if last batch is not full
        remaining_samples = count % self.batch_size

        if remaining_samples > 0:
            samples = self.gen.choice(self.data, size=remaining_samples)
            self.data = np.concatenate([self.data, samples], axis=0)
            
            # determine weights
            y = samples[:, self.indices['label']]
            w = np.ones((y.shape[0], 1), dtype=np.float32)
            
            w[y == 0.0] = w_b
            weights.append(w)
        
        self.weights = np.concatenate(weights, axis=0)

    def __len__(self):
        return int(self.data.shape[0] // self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        stop_idx = start_idx + self.batch_size

        z = self.data[start_idx:stop_idx]
        w = self.weights[start_idx:stop_idx]

        # split data (features, m, label)
        x = z[:, :self.indices['features']]
        m = z[:, self.indices['mass']].reshape(-1, 1)
        y = z[:, self.indices['label']].reshape(-1, 1)
        
        return dict(x=x, m=m), y, w

    def to_tf_dataset(self):
        return utils.dataset_from_sequence(self, sample_weights=True)


class BalancedSequence(tf.keras.utils.Sequence):
    """keras.Sequence that balances the signal (each m has the same number of events), and background"""

    def __init__(self, data, signal: pd.DataFrame, bkg: pd.DataFrame, batch_size: int, features: list, 
                 mass: list = None, all_bkg=False, seed=utils.SEED):
        assert batch_size >= 1
        
        mA = data.columns['mA']
        columns = features + [mA, data.columns['label']]
        
        if isinstance(mass, (list, np.ndarray)):
            self.mass = np.sort(mass)
        else:
            self.mass = data.unique_signal_mass
        
        self.gen = utils.get_random_generator(seed)
        
        # select data
        self.bkg = bkg[columns].values
        self.signals = {m: signal[signal[mA] == m][columns].values for m in self.mass}
        
        # batch-size for signal and background
        self.sig_batch = (batch_size / 2) // len(self.mass)
        self.bkg_batch = self.sig_batch * len(self.mass)
        
        # effective batch-size
        self.num_batches = int((signal.shape[0] + bkg.shape[0]) // (self.bkg_batch * 2))
            
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        # sample signal for each mA
        z = [np_sample(sig, amount=self.sig_batch, generator=self.gen) for sig in self.signals.values()]
        
        # sample from all bkg
        z.append(np_sample(self.bkg, amount=self.bkg_batch, generator=self.gen))
        
        z = np.concatenate(z, axis=0)

        # split data
        x = z[:, :-2]
        m = z[:, -2]
        y = z[:, -1]
        
        # sample mass (from signal's mA) for background events -> uses all background events
        mask = y == 0.0
        m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True)
        
        # reshape
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        return dict(x=x, m=m), y
    
    def to_tf_dataset(self):
        return utils.dataset_from_sequence(self)
    
    @classmethod
    def get_data(cls, dataset, train_batch=1024, eval_batch=1024, **kwargs):
        # split data
        train, valid = train_valid_split(dataset)
        
        # create sequences
        features = kwargs.pop('features', dataset.columns['feature'])
        
        train_seq = cls(dataset, signal=train[0], bkg=train[1], batch_size=int(train_batch),
                        features=features, **kwargs)
        
        valid_seq = EvalSequence(dataset, signal=valid[0], bkg=valid[1], batch_size=int(eval_batch),
                                 features=features, **kwargs)
        # return tf.Datasets
        return [seq.to_tf_dataset() for seq in [train_seq, valid_seq]]


class UniformSequence(BalancedSequence):
    """keras.Sequence that assigns a uniform mA for background"""
    
    def __init__(self, *args, interval=(100, 2000), **kwargs):
        super().__init__(*args, **kwargs)
        
        # sampling interval for mA
        self.interval = interval
    
    def __getitem__(self, idx):
        # sample signal for each mA
        z = [np_sample(sig, amount=self.sig_batch, generator=self.gen) for sig in self.signals.values()]
        
        # sample from all bkg
        z.append(np_sample(self.bkg, amount=self.bkg_batch, generator=self.gen))
        
        z = np.concatenate(z, axis=0)

        # split data
        x = z[:, :-2]
        m = z[:, -2]
        y = z[:, -1]
        
        # sample mass (from signal's mA) for background events -> uses all background events
        mask = y == 0.0
        m[mask] = self.gen.uniform(low=self.interval[0], high=self.interval[1], size=np.sum(mask))
        
        # reshape
        m = np.reshape(m, newshape=(-1, 1))
        y = np.reshape(y, newshape=(-1, 1))

        return dict(x=x, m=m), y


class TrainingSequence(tf.keras.utils.Sequence):
    """A simple training sequence for HEPMASS that allows sampling the mass feature"""

    def __init__(self, signal: pd.DataFrame, background: pd.DataFrame, columns: dict, batch_size: int,
                 features: list, seed=utils.SEED, sample_mass=False, uniform_mass: tuple = None, **kwargs):
        assert batch_size >= 1

        # retrieve columns
        mA = columns['mA']
        columns = features + [mA, columns['label']]

        self.indices = {'features': -2, 'mass': -2, 'label': -1}
        self.batch_size = int(batch_size)
        self.gen = utils.get_random_generator(seed)

        self.should_sample_mass = bool(sample_mass)
        self.should_sample_uniformly = isinstance(uniform_mass, tuple)
        self.mass_range = uniform_mass

        # mass & data
        self.mass = np.sort(signal[mA].unique())
        self.data = np.concatenate([signal[columns].values,
                                    background[columns].values], axis=0)
        self.gen.shuffle(self.data)

        # sample some datapoints if last batch is not full
        remaining_samples = len(self.data) % self.batch_size

        if remaining_samples > 0:
            samples = self.gen.choice(self.data, size=remaining_samples)
            self.data = np.concatenate([self.data, samples], axis=0)

        if self.should_sample_uniformly:
            # assign uniform mA to background
            mask = self.data[:, self.indices['label']] == 0

            m = self.data[:, self.indices['mass']]
            m[mask] = self.gen.uniform(low=self.mass_range[0], high=self.mass_range[1], 
                                       size=np.sum(mask))

            self.data[:, self.indices['mass']] = m
            print(self.data[self.gen.choice(np.arange(len(self.data)), size=16), self.indices['mass']])

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        stop_idx = start_idx + self.batch_size

        z = self.data[start_idx:stop_idx]

        # split data (features, mass, label)
        x = z[:, :self.indices['features']]
        m = z[:, self.indices['mass']].reshape(-1, 1)
        y = z[:, self.indices['label']].reshape(-1, 1)
        
        # sample mA for background
        if self.should_sample_mass:
            mask = y == 0.0

            if self.should_sample_uniformly:
                # uniform sampling
                m[mask] = self.gen.uniform(low=self.mass_range[0], high=self.mass_range[1], size=np.sum(mask))
            else:
                # identical sampling
                m[mask] = self.gen.choice(self.mass, size=np.sum(mask), replace=True)
        
        return dict(x=x, m=m), y

    def to_tf_dataset(self):
        return utils.dataset_from_sequence(self, sample_weights=False)

    @classmethod
    def get_data(cls, dataset, train_batch=1024, eval_batch=1024, **kwargs):
        # split data
        train, valid = train_valid_split(dataset, seed=kwargs.get('seed', utils.SEED))
        
        # create sequences
        features = kwargs.pop('features', dataset.columns['feature'])
        
        train_seq = cls(signal=train[0], background=train[1], columns=dataset.columns, 
                        batch_size=int(train_batch), features=features, **kwargs)
        
        kwargs.pop('sample_mass', None)
        kwargs.pop('uniform_mass', None)

        valid_seq = cls(signal=valid[0], background=valid[1], columns=dataset.columns, 
                        batch_size=int(eval_batch), features=features, sample_mass=False,
                        uniform_mass=None, **kwargs)

        # return tf.Datasets
        return [seq.to_tf_dataset() for seq in [train_seq, valid_seq]]

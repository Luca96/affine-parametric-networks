import os
import numpy as np
import pandas as pd

from script import SEED, free_mem

from typing import Union


class Hepmass:
    TRAIN_PATH = os.path.join('data', 'paper', 'all_train.csv')
    TEST_PATH = os.path.join('data', 'paper', 'all_test.csv')

    def __init__(self, x_scaler=None, m_scaler=None):
        self.ds = None
        self.columns = []

        self.x_scaler = x_scaler
        self.m_scaler = m_scaler
        
        self.features = None
        self.labels = None
        self.masses = None
        self.unique_mass = None
        
    def load(self, path: str, mass: Union[np.ndarray, tuple, list] = None, test_size=0.2, 
             fit_scaler=True, seed=SEED):
        """Loads the dataset:
            - selects feature columns,
            - scales the data if a sklearn.Scaler was provided,
            - splits all the data into train and test sets,
            - allows to select which mass to keep.
        """
        if self.ds is not None:
            return

        print('loading...')
        self.ds = pd.read_csv(path, dtype=np.float32, na_filter=False)
        
        # select columns
        columns = dict(feature=self.ds.columns[1:-1],
                       label=self.ds.columns[0], mass=self.ds.columns[-1])
        self.columns = columns
        
        # drop some mass
        if isinstance(mass, (list, tuple)):
            print('selecting mass...')
            self._select_mass(mass)
            free_mem()

        # select series
        self.features = self.ds[columns['feature']]
        self.labels = self.ds[columns['label']]
        self.masses = self.ds[columns['mass']]
        self.unique_mass = sorted(self.ds.mass.unique())
        
        # fit scaler
        if fit_scaler and (self.x_scaler is not None):
            self.x_scaler.fit(self.features.values)
        
        if fit_scaler and (self.m_scaler is not None):
            self.m_scaler.fit(np.reshape(self.unique_mass, newshape=(-1, 1)))
        
        print('dataset loaded.')
        free_mem()
    
    def get(self, mask=None) -> tuple:
        if mask is not None:
            features = self.features[mask].values
            labels = self.labels[mask].values
            mass = self.masses[mask].values
        else:
            features = self.features.values
            labels = self.labels.values
            mass = self.masses.values

        mass = mass.reshape((-1, 1))
        labels = labels.reshape((-1, 1))

        if self.x_scaler is not None:
            features = self.x_scaler.transform(features)
        
        if self.m_scaler is not None:
            mass = self.m_scaler.transform(mass)

        x = dict(x=features, m=mass)
        y = labels
        
        free_mem()
        return x, y
    
    def get_by_mass(self, mass: float) -> dict:
        return self.get(mask=self.ds.mass == mass)
    
    def _select_mass(self, mass):
        """Selects only the given mass from the dataframe"""
        for m in mass:
            # +/- 1 interval is used to account for floating-point inaccuracies
            mask = (self.ds.mass >= m - 1.0) & (self.ds.mass < m + 1.0)

            self.ds.drop(index=self.ds[mask].index, inplace=True)

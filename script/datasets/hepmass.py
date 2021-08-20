import os
import numpy as np
import pandas as pd

from script import free_mem
from script import utils

from typing import Union


class Hepmass:
    """Class that wraps the HEPMASS dataset utilized by Baldi et al. 2016"""

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
        
    def load(self, path: str, drop_mass: Union[np.ndarray, tuple, list] = None, 
             fit_scaler=True, robust=False, seed=utils.SEED):
        """Loads the dataset:
            - `drop_mass`: drops the provided mass values,
            - selects feature columns,
            - scales the data if a sklearn.Scaler was provided,
            - splits all the data into train and test sets,
            - allows to select which mass to keep.
            - if `robust=True`, outliers are clipped within [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
        """
        self.seed = seed
        
        if self.ds is not None:
            return

        print('loading...')
        self.ds = pd.read_csv(path, dtype=np.float32, na_filter=False)
        
        # select columns
        columns = dict(feature=self.ds.columns[1:-1],
                       label=self.ds.columns[0], mass=self.ds.columns[-1])
        self.columns = columns
        
        # drop some mass
        if isinstance(drop_mass, (list, tuple)):
            print('Removing masses...')
            self._remove_mass(drop_mass)
            free_mem()

        self.unique_mass = sorted(self.ds.mass.unique())

        if robust:
            print('clipping outliers..')
            self._clip_outliers()

        # select series
        self.features = self.ds[columns['feature']]
        self.labels = self.ds[columns['label']]
        self.masses = self.ds[columns['mass']]
        
        # fit scaler
        if fit_scaler and (self.x_scaler is not None):
            print('fitting feature scaler..')
            self.x_scaler.fit(self.features.values)
        
        if fit_scaler and (self.m_scaler is not None):
            print('fitting mass scaler..')
            self.m_scaler.fit(np.reshape(self.unique_mass, newshape=(-1, 1)))
        
        print('dataset loaded.')
        free_mem()
    
    def get(self, sample=None, mask=None) -> tuple:
        if mask is not None:
            features = self.features[mask]
            labels = self.labels[mask]
            mass = self.masses[mask]
        else:
            features = self.features
            labels = self.labels
            mass = self.masses

        # sample a subset of the selected data
        if isinstance(sample, float):
            features = features.sample(frac=sample, random_state=self.seed)
            labels = labels.loc[features.index]
            mass = mass.loc[features.index]

        features = features.values
        mass = mass.values.reshape((-1, 1))
        labels = labels.values.reshape((-1, 1))

        if self.x_scaler is not None:
            features = self.x_scaler.transform(features)
        
        if self.m_scaler is not None:
            mass = self.m_scaler.transform(mass)

        x = dict(x=features, m=mass)
        y = labels
        
        free_mem()
        return x, y
    
    def get_by_mass(self, mass: float, sample=None) -> dict:
        return self.get(mask=self.ds.mass == mass, sample=sample)
    
    def scale_mass(self, mass) -> np.ndarray:
        if self.m_scaler is None:
            return mass

        mass = np.reshape(mass, newshape=(-1, 1))
        mass = self.m_scaler.transform(mass)
        return np.squeeze(mass)

    def _remove_mass(self, mass):
        """Removes only the given mass from the dataframe"""
        for m in mass:
            # +/- 1 interval is used to account for floating-point inaccuracies
            mask = (self.ds.mass >= m - 1.0) & (self.ds.mass < m + 1.0)

            self.ds.drop(index=self.ds[mask].index, inplace=True)

    def _clip_outliers(self):
        for mass in self.unique_mass:
            for col in self.columns['feature']:
                for label in [0.0, 1.0]:
                    mask = (self.ds['mass'] == mass) & (self.ds[self.columns['label']] == label)
                    
                    # select data
                    serie = self.ds.loc[mask, col]
                    
                    # get quantiles
                    # source: https://paolapozzolo.it/boxplot/
                    stats = serie.describe()
                    
                    q1 = stats['25%']
                    q3 = stats['75%']
                    iqr = q3 - q1  # inter-quartile range
                    
                    low = q1 - 1.5 * iqr
                    upp = q3 + 1.5 * iqr
                    
                    # clip
                    serie.clip(lower=low, upper=upp, inplace=True)
                    
                    # apply changes
                    self.ds.loc[mask, col] = serie

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split

from script import free_mem, assert_2d_array
from script import utils

from typing import Union, List


class Dataset:
    """Class that wraps Tommaso's MC data"""

    SIGNAL_PATH = os.path.join('data', 'signal.csv')
    BACKGROUND_PATH = os.path.join('data', 'background.csv')
    
    FEATURE_COLUMNS = [
        "dimuon_deltar", 
        "dimuon_deltaphi", 
        "dimuon_deltaeta", 
        "met_pt", 
        "deltar_bjet1_dimuon", 
        "deltapt_bjet1_dimuon", 
        "deltaeta_bjet1_dimuon", 
        "bjet_1_pt", 
        "bjet_1_eta", 
        "ljet_1_pt", 
        "ljet_1_eta", 
        "bjet_n", 
        "ljet_n"]
    
    MASS_INTERVALS = [
        (105, 115),  # 10 wide
        (115, 125),
        (125, 135),
        (135, 145),
        (145, 155),
        (155, 165),
        (165, 175),
        (175, 185),
        (185, 195),
        (195, 205),
        (212.5, 237.5), # 25 wide
        (237.5, 262.5),
        (262.5, 287.5),
        (287.5, 312.5),
        (325, 375),  # 50 wide
        (375, 425),
        (425, 475),
        (475, 525),  # 100 wide
        (550, 650),
        (650, 750),
        (750, 850),
        (850, 950),
        (950, 1050)]
    
    def __init__(self, x_scaler=None, m_scaler=None, seed=utils.SEED):
        self.seed = seed
        
        self.signal = None
        self.background = None
        
        self.ds = None
        self.columns = []
        self.train_df = None
        self.test_df = None
        
        self.unique_signal_mass = None
        self.current_mass_intervals = None
        self.signal_mass_bins = None
        self.changed_bkg_mass = False
        
        self.num_classes = None
        self.mass_to_label = None
        
        self.x_scaler = x_scaler
        self.m_scaler = m_scaler
        
        self.train_features = None
        self.test_features = None
        self.train_labels = None
        self.test_labels = None
        self.train_mass = None
        self.test_mass = None
        
        # keep sample weights
        self.weights_df = None
    
    def load(self, mass_intervals: Union[np.ndarray, List[tuple]] = None, test_size=0.2, change_bkg_mass=False,
             feature_columns=None, robust=False, multi_class=False):
        """Loads the signal+background data:
            - selects feature columns,
            - scales the data if a sklearn.Scaler was provided,
            - splits all the data into train and test sets,
            - allows to select which mass to keep: `mass_intervals` is an array  of shape (N, 2) which 
              specifies a "lower" and "upper" interval for the mass.
            - converts the problem to multi-class classification, if `multi_class` is True.
        """
        assert_2d_array(mass_intervals)
        
        if self.ds is not None:
            return

        print('[signal] loading...')
        self.signal = pd.read_csv(self.SIGNAL_PATH, dtype=np.float32, na_filter=False)
        
        print('[background] loading...')
        self.background = pd.read_csv(self.BACKGROUND_PATH, dtype=np.float32, na_filter=False)
        
        self.ds = pd.concat([self.signal, self.background])
        
        # mass intervals
        self.unique_signal_mass = sorted(self.signal['mA'].unique())
        
        if mass_intervals is not None:
            self.current_mass_intervals = mass_intervals
        else:
            self.current_mass_intervals = self.MASS_INTERVALS
        
        self.signal_mass_bins = [x for x, _ in self.current_mass_intervals] + [1050]  # for `tfp.stats.find_bins` only
        self.signal_mass_bins = tf.constant(self.signal_mass_bins, dtype=tf.float32)
        
        # bkg's mass is uniformly distributed, make it distributes as signal's mass
        if change_bkg_mass:
            print('[Dataset] setting bkg = sig...')
            self._bkg_mass_as_signal()
            self.changed_bkg_mass = True

            # last mass interval is apparently lost
            self.current_mass_intervals = self.current_mass_intervals[:-1]
        
        # from binary to multi-class classification
        if multi_class:
            print('[Dataset] making multi-class labels...')
            self._make_multi_class()
        
        # select columns
        if feature_columns == 'all':
            feature_columns = self.ds.columns[1:-3]
            
        elif not isinstance(feature_columns, list):
            feature_columns = self.FEATURE_COLUMNS  # select default features
        
        columns = dict(feature=feature_columns,
                       weight=['weight', 'PU_Weight'],
                       label=self.ds.columns[-1], mass=self.ds.columns[0])
        self.columns = columns
        
        # remove outliers
        if robust:
            print('[Dataset] clipping outliers..')
            self._clip_outliers()
        
        # train-test split:
        self.train_df, self.test_df = train_test_split(self.ds, test_size=test_size, 
                                                       random_state=self.seed)
        # fit scaler (on "whole" training-set)
        if self.x_scaler is not None:
            print('[Dataset] fitting feature scaler..')
            self.x_scaler.fit(self.train_df[columns['feature']].values)
        
        if self.m_scaler is not None:
            print('[Dataset] fitting mass scaler..')
            self.m_scaler.fit(self.train_df[columns['mass']].unique().reshape((-1, 1)))
        
        # drop some mass
        if isinstance(mass_intervals, (list, np.ndarray)):
            print('[Dataset] selecting mass-intervals...')
            self._select_mass(intervals=mass_intervals)
            
            self.train_df, self.test_df = train_test_split(self.ds, test_size=test_size,
                                                           random_state=self.seed)
            free_mem()

        # select series
        self.train_features = self.train_df[columns['feature']]
        self.train_labels = self.train_df[columns['label']]
        self.train_mass = self.train_df[columns['mass']]
        
        self.test_features = self.test_df[columns['feature']]
        self.test_labels = self.test_df[columns['label']]
        self.test_mass = self.test_df[columns['mass']]
        
        # select "sample weights" for training data only
        self.weights_df = self.train_df[columns['weight']]
        
        print('[Dataset] loaded.')
        free_mem()
    
    def get(self, mask=None, sample=None, transformer=None) -> tuple:
        if isinstance(mask, str) and (mask == 'test'):
            features = self.test_features
            labels = self.test_labels
            mass = self.test_mass
            
        elif mask is not None:
            features = self.test_features[mask]
            labels = self.test_labels[mask]
            mass = self.test_mass[mask]
        else:
            features = self.train_features
            labels = self.train_labels
            mass = self.train_mass
        
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
    
        if transformer is not None:
            features = transformer.transform(features)
    
        x = dict(x=features, m=mass)
        
        if self.num_classes is not None:
            labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
        
        free_mem()
        return x, labels
    
    def get_by_mass(self, interval: Union[list, tuple], sample=None, transformer=None) -> dict:
        mass_low, mass_high = interval
        
        return self.get(mask=(self.test_mass >= mass_low) & (self.test_mass < mass_high), sample=sample,
                        transformer=transformer)
    
    def mass_to_categories(self, mass: list):
        """Maps each given mass into an iterval, i.e. retrieving its index to be used as category"""
        bins = tfp.stats.find_bins(tf.cast(mass, dtype=tf.float32), self.signal_mass_bins)

        return tf.keras.utils.to_categorical(bins, num_classes=len(self.unique_signal_mass))
    
    def _select_mass(self, intervals: Union[list, np.ndarray]):
        """Selects only the given mass from the dataframe"""
        mask = np.full(shape=(len(self.ds),), fill_value=False)
        
        # get at which index there are the requested mass
        for (m_low, m_upp) in intervals:
            mask |= (self.ds['mA'] >= m_low) & (self.ds['mA'] < m_upp)

        # select data
        self.ds = self.ds[mask]
        free_mem()
    
    def _bkg_mass_as_signal(self):
        bkg_mass = self.ds[self.ds['type'] == 0.0]['mA']
        sig_mass = self.unique_signal_mass + [1100.0]
        
        for i in range(len(sig_mass) - 1):
            m_low = sig_mass[i]
            m_upp = sig_mass[i + 1]
            
            mask = (bkg_mass >= m_low) & (bkg_mass < m_upp)
            self.ds.loc[mask, 'mA'] = m_low
    
    def _make_multi_class(self):
        # build a dict: mass-interval -> label
        self.interval_to_label = {m: i for i, m in enumerate(self.current_mass_intervals)}
        num_intervals = len(self.current_mass_intervals)
        
        # locate signal entries
        signal_mask = self.ds['type'] == 1.0
        bkg_mask = self.ds['type'] == 0.0
        
        # change labels
        for (m_low, m_upp), label in self.interval_to_label.items():
            mask = (self.ds['mA'] >= m_low) & (self.ds['mA'] < m_upp)
            
            self.ds.loc[mask & signal_mask, 'type'] = label + num_intervals
            self.ds.loc[mask & bkg_mask, 'type'] = label
            
        self.num_classes = num_intervals * 2

    def _clip_outliers(self):
        for (low, upp) in self.current_mass_intervals:
            for col in self.columns['feature']:
                for label in [0.0, 1.0]:
                    mask = (self.ds['mA'] >= low) & (self.ds['mA'] < upp) & \
                           (self.ds[self.columns['label']] == label)
                    
                    # select data
                    serie = self.ds.loc[mask, col]
                    
                    # get quantiles
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

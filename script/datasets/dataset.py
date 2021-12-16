import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import enum

from sklearn.model_selection import train_test_split

from script import free_mem, assert_2d_array
from script import utils

from typing import Union, List


class Dataset:
    """Class that wraps Tommaso's MC data"""

    # TODO: edit default paths
    SIGNAL_PATH = os.path.join('data', 'signal.csv')
    BACKGROUND_PATH = os.path.join('data', 'background.csv')
    BACKGROUND_PATH2 = os.path.join('data', 'mcs', 'background_physweight2.csv')
    
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
    
    # TODO: remove all the unused methods
    def load(self, signal: Union[str, list, pd.DataFrame] = None, bkg: Union[str, list, pd.DataFrame] = None, test_size=0.2, 
             mass_intervals: Union[np.ndarray, List[tuple]] = None, change_bkg_mass=False, feature_columns=None, mass_column='mA',
             multi_class=False, sample_bkg=False, add_var=False):
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

        # loading SIGNAL
        print('[signal] loading...')

        self.signal = self._load_csv(signal, default_path=self.SIGNAL_PATH)
        self.signal['name'] = 'signal'

        # loading BACKGROUND
        print('[background] loading...')

        self.background = self._load_csv(bkg, default_path=self.BACKGROUND_PATH2)

        if 'bkg_name' in self.background.columns:
            def convert(x):
                if 'ST_' in x:
                    return 'ST'
                
                if 'TTbar' in x:
                    return 'TTbar'

                if 'diboson_' in x:
                    return 'diboson'

                # if 'ZMM_' in x:
                #     return 'DY'

                if 'ZMM_' in x:
                    return 'ZMM'
                
                return x

            self.background['name'] = self.background['bkg_name'].apply(convert)
            self.names_df = pd.DataFrame({'name': self.background['name']})
            self.original_names = pd.DataFrame({'name': self.background['bkg_name']})
            self.background.drop(columns=['bkg_name'], inplace=True)

        # add new var
        if add_var:
            self.signal['dimuon_pt_M'] = self.signal['dimuon_pt'] / self.signal['dimuon_M']
            self.background['dimuon_pt_M'] = self.background['dimuon_pt'] / self.background['dimuon_M']

        self.ds = pd.concat([self.signal, self.background], ignore_index=True)
        # DO NOT reset index
        
        # mass intervals
        self.unique_signal_mass = np.sort(self.signal['mA'].unique())

        assert mass_column in self.ds.columns
        self.mass_column = mass_column
        
        if mass_intervals is not None:
            self.current_mass_intervals = mass_intervals
        else:
            self.current_mass_intervals = self.MASS_INTERVALS
        
        self.signal_mass_bins = [x for x, _ in self.current_mass_intervals] + [self.ds['dimuon_M'].max() + 1.0]  # for `tfp.stats.find_bins` only
        self.signal_mass_bins = tf.constant(self.signal_mass_bins, dtype=tf.float32)
        
        # bkg's mass is uniformly distributed, make it distributes as signal's mass
        if change_bkg_mass:
            print('[Dataset] setting bkg = sig...')
            self._bkg_mass_as_signal()
            self.changed_bkg_mass = True

            # last mass interval is apparently lost
            self.current_mass_intervals = self.current_mass_intervals[:-1]
        
        if sample_bkg:
            self.should_sample_bkg = True

            if self.m_scaler is None:
                self.mass_for_bkg = self.unique_signal_mass
            else:
                mass = self.scale_mass(self.unique_signal_mass)
                self.mass_for_bkg = np.squeeze(mass)
        else:
            self.should_sample_bkg = False

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
                       label='type', mass='mA')
        self.columns = columns
        
        # add new feature: "dimuon_pt / dimuon_M"
        if add_var:
            self.columns['feature'].append('dimuon_pt_M')

        # # remove outliers
        # if robust:
        #     print('[Dataset] clipping outliers..')
        #     self._clip_outliers()
    
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
        self.train_mass_col = self.train_df[self.mass_column]
        
        self.test_features = self.test_df[columns['feature']]
        self.test_labels = self.test_df[columns['label']]
        self.test_mass = self.test_df[columns['mass']]
        self.test_mass_col = self.test_df[self.mass_column]
        
        # select "sample weights" for training data only
        self.weights_df = self.train_df[columns['weight']]
        
        print('[Dataset] loaded.')
        free_mem()
    
    def get(self, mask=None, sample=None, transformer=None, split=None) -> tuple:
        if isinstance(split, str):
            split = split.lower()
    
        if isinstance(mask, str) and (mask == 'test'):
            features = self.test_features
            labels = self.test_labels
            mass = self.test_mass
            
        elif mask is not None:
            if split in ['test', None]:
                features = self.test_features[mask]
                labels = self.test_labels[mask]
                mass = self.test_mass[mask]
            else:
                features = self.train_features[mask]
                labels = self.train_labels[mask]
                mass = self.train_mass[mask]
        else:
            if split == 'test':
                features = self.test_features
                labels = self.test_labels
                mass = self.test_mass
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
        
        if self.num_classes is not None:
            labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
        
        if self.should_sample_bkg:
            mask = np.squeeze(labels == 0.0)

            if np.sum(mask) > 0:
                bkg_mass = np.random.choice(self.mass_for_bkg, size=np.sum(mask), replace=True)
                mass[mask] = np.reshape(bkg_mass, newshape=(-1, 1))

        x = dict(x=features.astype(np.float32), m=mass.astype(np.float32))

        free_mem()
        return x, labels.astype(np.float32)
    
    def get_by_mass(self, interval: Union[list, tuple], sample=None, transformer=None, split='test',
                    interval_as_mass=False) -> tuple:
        mass_low, mass_high = interval
        
        if split == 'all':
            x_test, y_test = self.get_by_mass(interval, sample=sample, split='test')
            x_train, y_train = self.get_by_mass(interval, sample=sample, split='train')

            x = {k: np.concatenate([x_train[k], v], axis=0) for k, v in x_test.items()}
            y = np.concatenate([y_train, y_test], axis=0)

            return x, y

        elif split == 'test':
            mass = self.test_mass_col
        else:
            mass = self.train_mass_col

        x, y = self.get(mask=(mass >= mass_low) & (mass < mass_high), sample=sample,
                        transformer=transformer, split=split)

        if interval_as_mass:
            mask = np.squeeze(y == 0.0)  # bkg
            x['m'][mask] = self.scale_mass(np.mean(interval))

        return x, y
    
    def get_and_change_mass(self, interval: Union[list, tuple], mass: float, sample=None) -> tuple:
        mass_low, mass_high = interval
        mass_mask = (self.test_mass >= mass_low) & (self.test_mass < mass_high)

        sig_mask = mass_mask & (self.test_labels == 1.0)
        bkg_mass = mass_mask & (self.test_labels == 0.0)

        x, labels = self.get(mask=sig_mask | bkg_mask, sample=sample)
        x['m'][sig_mask] = mass

        return x, labels

    def scale_mass(self, mass) -> np.ndarray:
        if self.m_scaler is None:
            return mass

        mass = np.reshape(mass, newshape=(-1, 1))
        mass = self.m_scaler.transform(mass)
        return np.squeeze(mass)

    def mass_to_categories(self, mass: list):
        """Maps each given mass into an iterval, i.e. retrieving its index to be used as category"""
        bins = tfp.stats.find_bins(tf.cast(mass, dtype=tf.float32), self.signal_mass_bins)

        return tf.keras.utils.to_categorical(bins, num_classes=len(self.unique_signal_mass))
    
    def _load_csv(self, csv: Union[str, list, pd.DataFrame], default_path: str, dtype=np.float32) -> pd.DataFrame:
        if csv is None or isinstance(csv, str):
            return self._safe_convert(pd.read_csv(csv or default_path, dtype=None, na_filter=False), dtype)
        
        elif isinstance(csv, pd.DataFrame):
            return self._safe_convert(csv, dtype)

        elif isinstance(csv, (list, tuple)):
            return pd.concat([self._load_csv(x, default_path, dtype) for x in csv], ignore_index=True)
        else:
            raise ValueError

    def _safe_convert(self, df: pd.DataFrame, dtype=np.float32):
        # convert only columns that are not "object" nor "string"
        mask = (df.dtypes != 'object') & (df.dtypes != 'string')
        columns = list(df.dtypes[mask].index)

        df[columns] = df[columns].astype(dtype, copy=False)
        return df 

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

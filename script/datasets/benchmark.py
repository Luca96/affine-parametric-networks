import os
import numpy as np
import pandas as pd

from script import utils
from typing import Union


class Benchmark:
    """Class that wraps our HEPMASS-IMB benchmark dataset"""
    
    def __init__(self):
        self.ds = None
        self.signal = None
        self.background = None
        
        self.columns = None
        self.unique_signal_mass = None
    
    def __len__(self):
        return len(self.ds)
    
    def load(self, signal: Union[str, list, pd.DataFrame] = None, features: list = None,
             bkg: Union[str, list, pd.DataFrame] = None, mass_intervals: list = None):
        """Loads the dataset"""
        # loading SIGNAL
        print('[signal] loading...')

        self.signal = self._load_csv(signal)
        self.signal['name'] = 'signal'

        # loading BACKGROUND
        print('[background] loading...')
        
        self.background = self._load_csv(bkg)
        self.names_df = pd.DataFrame({'name': self.background['name']})

        # TODO: if unused delete
        # concatenate
        self.ds = pd.concat([self.signal, self.background], ignore_index=True)
        # DO NOT reset index
        
        # select columns
        self.columns = dict(feature=features, mass='mass', label='type')

        # mass
        self.unique_signal_mass = np.sort(self.signal['mass'].unique())

        # mass intervals
        self.default_intervals = np.array([(-np.inf, np.inf)] * len(self.unique_signal_mass))
        
        if mass_intervals is None:
            self.mass_intervals = self.default_intervals.copy()  # 1-vs-all intervals
        else:
            assert isinstance(mass_intervals, (np.ndarray, list))
            self.mass_intervals = np.array(mass_intervals)
        
        print('dataset loaded.')
        utils.free_mem()
    
    def _load_csv(self, csv: Union[str, list, pd.DataFrame], dtype=np.float32) -> pd.DataFrame:
        if isinstance(csv, str):
            return self._safe_convert(pd.read_csv(csv, dtype=None, na_filter=False), dtype)
        
        elif isinstance(csv, pd.DataFrame):
            return self._safe_convert(csv, dtype)

        elif isinstance(csv, (list, tuple)):
            return pd.concat([self._load_csv(x, dtype) for x in csv],
                             ignore_index=True)
        else:
            raise ValueError('Provide path (str), pd.DataFrame or list.')

    def _safe_convert(self, df: pd.DataFrame, dtype=np.float32):
        # convert only columns that are not "object" nor "string"
        mask = (df.dtypes != 'object') & (df.dtypes != 'string')
        columns = list(df.dtypes[mask].index)

        df[columns] = df[columns].astype(dtype, copy=False)
        return df


# to ensure compatibility with old code
class Dataset(Benchmark):
    FEATURE_COLUMNS = []
    MASS_INTERVALS = []

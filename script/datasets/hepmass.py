import os
import numpy as np
import pandas as pd

from script import utils


class Hepmass:
    """Class that wraps the HEPMASS dataset utilized by Baldi et al. 2016"""

    TRAIN_PATH = os.path.join('data', 'hepmass', 'train.csv')
    TEST_PATH = os.path.join('data', 'hepmass', 'test.csv')

    def __init__(self):
        self.ds = None
        self.signal = None
        self.background = None
        
        self.columns = None
        self.unique_signal_mass = None
        
    def load(self, path: str, signal: pd.DataFrame = None, bkg: pd.DataFrame = None, 
             mass_intervals: list = None):
        """Loads the dataset"""
        print('loading...')
        
        if isinstance(path, str):
            # if path is provided, load csv from disk
            self.ds = pd.read_csv(path, dtype=np.float32, na_filter=False)

            self.signal = self.ds[self.ds['type'] == 1]
            self.background = self.ds[self.ds['type'] == 0]
        else:
            # else, must provide two dataframes (signal and bkg)
            assert isinstance(signal, pd.DataFrame) and isinstance(bkg, pd.DataFrame)
            
            self.signal = signal
            self.background = bkg
            
            self.ds = pd.concat([signal, bkg])
        
        # select columns
        self.columns = dict(feature=list(self.ds.columns[1:-1]), mA='mass',
                            label='type', mass='f26')
        # mass
        self.unique_signal_mass = np.sort(self.signal['mass'].unique())

        # mass intervals
        if mass_intervals is None:
            mass = self.ds['f26']
            
            self.mass_intervals = [(mass.min(), mass.max())] * len(self.unique_signal_mass)
            self.mass_intervals = np.array(self.mass_intervals)
        else:
            assert isinstance(mass_intervals, (np.ndarray, list))
            self.mass_intervals = np.array(mass_intervals)
        
        print('dataset loaded.')
        utils.free_mem()
    
    def to_dataset(self, batch_size: int, features: list = None, validation_split=0.25):
        assert batch_size >= 1
        
        if features is None:
            features = self.columns['feature']
        
        x = self.ds[features].values
        m = self.ds['mass'].values.reshape(-1, 1)
        y = self.ds['type'].values.reshape(-1, 1)
        
        train_ds, valid_ds = utils.dataset_from_tensors(tensors=({'x': x, 'm': m}, y),
                                                        batch_size=int(batch_size),
                                                        split=float(validation_split))
        utils.free_mem()
        return train_ds, valid_ds

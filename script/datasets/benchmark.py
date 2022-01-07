import os
import numpy as np
import pandas as pd

from script import utils


class Benchmark:
    """Class that wraps our BENCHMARK dataset"""
    
    FEATURES = ["dimuon_deltar", "dimuon_deltaphi", "dimuon_deltaeta", "met_pt", 
                 "deltar_bjet1_dimuon", "deltapt_bjet1_dimuon", "deltaeta_bjet1_dimuon", 
                 "bjet_1_pt", "bjet_1_eta", "deltaphi_bjet1_dimuon",
                 "ljet_1_pt", "ljet_1_eta", "bjet_n", "ljet_n"]
    
    DISJOINT_INTERVALS = np.array([
        (105, 115),      # 10 wide
        (115, 125),
        (125, 135),
        (135, 145),
        (145, 155),
        (155, 165),
        (165, 175),
        (175, 185),
        (185, 195),
        (195, 205),
        (212.5, 237.5),  # 25 wide
        (237.5, 262.5),
        (262.5, 287.5),
        (287.5, 312.5),
        (325, 375),      # 50 wide
        (375, 425),
        (425, 475),
        (475, 525),      # 100 wide
        (550, 650),
        (650, 750),
        (750, 850),
        (850, 950),
        (950, 1050)])
    
    # TODO: tune at beginning (shrink) and end (enlarge)
    INTERVALS = np.array([
        (100, 130),    # 110
        (115, 135),    # 120
        (120, 140),    # 130
        (130, 155),    # 140
        (140, 165),    # 150
        (150, 170),    # 160
        (160, 180),    # 170
        (170, 190),    # 180
        (180, 200),    # 190
        (185, 210),    # 200
        (200, 250),    # 225
        (225, 275),    # 250
        (250, 300),    # 275
        (250, 350),    # 300
        (270, 400),    # 350
        (320, 450),    # 400
        (340, 530),    # 450
        (350, 580),    # 500
        (350, 750),    # 600
        (400, 850),    # 700
        (500, 950),    # 800
        (550, 1100),   # 900
        (550, 1200)])  # 1000
    
    def __init__(self):
        self.ds = None
        self.signal = None
        self.background = None
        
        self.columns = None
        self.unique_signal_mass = None
        
    def load(self, path: str, signal: pd.DataFrame = None, bkg: pd.DataFrame = None, 
             mass_intervals: list = None, features: list = None):
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
        self.columns = dict(feature=features or Benchmark.FEATURES, mA='mA',
                            label='type', mass='dimuon_mass')
        # mass
        self.unique_signal_mass = np.sort(self.signal['mA'].unique())

        # mass intervals
        if mass_intervals is None:
            self.mass_intervals = Benchmark.DISJOINT_INTERVALS
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

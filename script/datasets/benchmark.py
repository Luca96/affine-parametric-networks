import os
import numpy as np
import pandas as pd

from script import utils
from typing import Union


class Benchmark:
    """Class that wraps our BENCHMARK dataset"""
    
    FEATURES = ["dimuon_deltar", "dimuon_deltaphi", "dimuon_deltaeta", "met_pt", 
                 "deltar_bjet1_dimuon", "deltapt_bjet1_dimuon", "deltaeta_bjet1_dimuon", 
                 "bjet_1_pt", "bjet_1_eta", "deltaphi_bjet1_dimuon",
                 "ljet_1_pt", "ljet_1_eta", "bjet_n", "ljet_n"]
    
    DISJOINT_INTERVALS = np.array([
        (100, 115),      
        (115, 125),      # 10 wide
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
        (950, np.inf)])
    
    INTERVALS = np.array([
        (100, 130),      # 110
        (115, 135),      # 120
        (120, 140),      # 130
        (130, 155),      # 140
        (140, 165),      # 150
        (150, 170),      # 160
        (160, 180),      # 170
        (170, 190),      # 180
        (180, 200),      # 190
        (185, 210),      # 200
        (200, 250),      # 225
        (225, 275),      # 250
        (250, 300),      # 275
        (250, 350),      # 300
        (270, 400),      # 350
        (320, 450),      # 400
        (340, 530),      # 450
        (350, 580),      # 500
        (350, 750),      # 600
        (400, 850),      # 700
        (500, 950),      # 800
        (550, 1100),     # 900
        (550, np.inf)])  # 1000
    
    # min-max values for variable clipping
    CLIP_RANGES = {'dimuon_deltar': (0.5, 5.0), 'dimuon_deltaphi': (0.01, np.inf),
                   'dimuon_deltaeta': (0, 4.5), 'dimuon_pt': (0, 750),
                   'met_pt': (0, 500), 'met_phi': (-np.inf, np.inf), 'bjet_n': (0, 4),
                   'bjet_1_pt': (0, 650), 'bjet_1_eta': (-np.inf, np.inf), 'jetfwd_n': (0, 5),
                   'ljet_n': (0, 9), 'ljet_1_pt': (-np.inf, 700), 'ljet_1_eta': (-np.inf, np.inf),
                   'deltar_bjet1_dimuon': (0.075, 7.5), 'deltapt_bjet1_dimuon': (0, 500),
                   'deltaeta_bjet1_dimuon': (0, 7.5), 'deltaphi_bjet1_dimuon': (-np.inf, np.inf)}
    
    # train-set statistics (on clipped variables)
    STATS = {
        "dimuon_deltar": {
            "min": 0.5,
            "max": 5.0,
            "mean": 2.8884029388427734,
            "std": 0.5713036060333252,
            "25%": 2.561470329761505,
            "75%": 3.2018219232559204,
        },
        "dimuon_deltaphi": {
            "min": 0.009999999776482582,
            "max": 3.1415863037109375,
            "mean": 2.4677369594573975,
            "std": 0.6579403877258301,
            "25%": 2.16943359375,
            "75%": 2.9648258686065674,
        },
        "dimuon_deltaeta": {
            "min": 0.0,
            "max": 4.5,
            "mean": 1.204237699508667,
            "std": 0.8349418640136719,
            "25%": 0.5189208984375,
            "75%": 1.75787353515625,
        },
        "dimuon_mass": {
            "min": 100.0003890991211,
            "max": 3566.62060546875,
            "mean": 319.4694519042969,
            "std": 252.1707763671875,
            "25%": 134.66471099853516,
            "75%": 426.9085998535156,
        },
        "dimuon_pt": {
            "min": 0.11833428591489792,
            "max": 750.0,
            "mean": 84.40518951416016,
            "std": 64.79308319091797,
            "25%": 43.32844638824463,
            "75%": 105.64176750183105,
        },
        "met_pt": {
            "min": 0.058971043676137924,
            "max": 500.0,
            "mean": 59.19718933105469,
            "std": 46.014957427978516,
            "25%": 26.924589157104492,
            "75%": 79.1959171295166,
        },
        "met_phi": {
            "min": -3.1416015625,
            "max": 3.1416015625,
            "mean": 0.0050789909437298775,
            "std": 1.8952653408050537,
            "25%": -1.71240234375,
            "75%": 1.723876953125,
        },
        "bjet_n": {
            "min": 1.0,
            "max": 4.0,
            "mean": 1.2298729419708252,
            "std": 0.4420945644378662,
            "25%": 1.0,
            "75%": 1.0,
        },
        "bjet_1_pt": {
            "min": 20.015625,
            "max": 650.0,
            "mean": 80.76116943359375,
            "std": 59.50202941894531,
            "25%": 40.21875,
            "75%": 101.9375,
        },
        "bjet_1_eta": {
            "min": -2.39990234375,
            "max": 2.39990234375,
            "mean": 0.00538437208160758,
            "std": 1.1365785598754883,
            "25%": -0.856689453125,
            "75%": 0.869140625,
        },
        "jetfwd_n": {
            "min": 0.0,
            "max": 5.0,
            "mean": 0.5440199971199036,
            "std": 0.7681567668914795,
            "25%": 0.0,
            "75%": 1.0,
        },
        "ljet_n": {
            "min": 0.0,
            "max": 9.0,
            "mean": 1.17392897605896,
            "std": 1.3013898134231567,
            "25%": 0.0,
            "75%": 2.0,
        },
        "ljet_1_pt": {
            "min": -1.0,
            "max": 700.0,
            "mean": 46.12085723876953,
            "std": 67.33338928222656,
            "25%": -1.0,
            "75%": 63.375,
        },
        "ljet_1_eta": {
            "min": -3.0,
            "max": 2.39990234375,
            "mean": -1.1679404973983765,
            "std": 1.7731586694717407,
            "25%": -3.0,
            "75%": 0.36492919921875,
        },
        "deltar_bjet1_dimuon": {
            "min": 0.07500000298023224,
            "max": 7.5,
            "mean": 2.929969310760498,
            "std": 1.0841649770736694,
            "25%": 2.2905060052871704,
            "75%": 3.4583510160446167,
        },
        "deltapt_bjet1_dimuon": {
            "min": 4.461943262867862e-06,
            "max": 500.0,
            "mean": 41.990211486816406,
            "std": 45.575782775878906,
            "25%": 12.126170635223389,
            "75%": 56.29130935668945,
        },
        "deltaeta_bjet1_dimuon": {
            "min": 3.035472445844789e-06,
            "max": 7.5,
            "mean": 1.5736356973648071,
            "std": 1.2492525577545166,
            "25%": 0.5985857397317886,
            "75%": 2.2482579946517944,
        },
        "deltaphi_bjet1_dimuon": {
            "min": 1.5612924471497536e-05,
            "max": 3.1415908336639404,
            "mean": 2.2390286922454834,
            "std": 0.8433079719543457,
            "25%": 1.7371848225593567,
            "75%": 2.9230599403381348,
        }}
    
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
        self.columns = dict(feature=features or Benchmark.FEATURES, mA='mA',
                            label='type', mass='dimuon_mass')
        # mass
        self.unique_signal_mass = np.sort(self.signal['mA'].unique())

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

import numpy as np
import pandas as pd
import tensorflow as tf

from script.datasets import Dataset


# constants
CLIP_CAT1 = {'dimuon_deltar': (0.5, 5.5), 'dimuon_deltaphi': (-np.inf, np.inf), 
             'dimuon_deltaeta': (-np.inf, np.inf), 'dimuon_pt': (-np.inf, np.inf), 
             'met_pt': (0.0, 500), 'met_phi': (-np.inf, np.inf), 'bjet_n': (-np.inf, np.inf), 
             'bjet_1_pt': (0.0, 800), 'bjet_1_eta': (-np.inf, np.inf), 
             'jetfwd_n': (-np.inf, np.inf), 'ljet_n': (-np.inf, np.inf), 'ljet_1_pt': (0.0, 1000), 
             'ljet_1_eta': (-np.inf, np.inf), 'deltar_bjet1_dimuon': (0.0, 7.5), 
             'deltapt_bjet1_dimuon': (0.0, 700), 'deltaeta_bjet1_dimuon': (0.0, 7), 
             'deltaphi_bjet1_dimuon': (-np.inf, np.inf)}

STATS_CAT1 = {
    "dimuon_deltar": {
        "mean": 2.875296115875244,
        "std": 0.584187924861908,
        "min": 0.5,
        "max": 5.5,
        "25%": 2.5123260021209717,
        "75%": 3.2037512063980103,
    },
    "dimuon_deltaphi": {
        "mean": 2.3938915729522705,
        "std": 0.6741409301757812,
        "min": 0.0,
        "max": 3.1415913105010986,
        "25%": 2.0644352436065674,
        "75%": 2.91650390625,
    },
    "dimuon_deltaeta": {
        "mean": 1.292601227760315,
        "std": 0.8675311207771301,
        "min": 0.0,
        "max": 4.7802734375,
        "25%": 0.5776329040527344,
        "75%": 1.87841796875,
    },
    "dimuon_pt": {
        "mean": 86.28317260742188,
        "std": 73.06474304199219,
        "min": 0.043848637491464615,
        "max": 6038.21484375,
        "25%": 43.80977916717529,
        "75%": 105.55172920227051,
    },
    "met_pt": {
        "mean": 63.94449996948242,
        "std": 46.91404342651367,
        "min": 0.026056507602334023,
        "max": 500.0,
        "25%": 30.1546049118042,
        "75%": 85.95086669921875,
    },
    "met_phi": {
        "mean": 0.017062433063983917,
        "std": 1.868918776512146,
        "min": -3.1416015625,
        "max": 3.1416015625,
        "25%": -1.655029296875,
        "75%": 1.68603515625,
    },
    "bjet_n": {
        "mean": 1.3534600734710693,
        "std": 0.506399929523468,
        "min": 1.0,
        "max": 6.0,
        "25%": 1.0,
        "75%": 2.0,
    },
    "bjet_1_pt": {
        "mean": 87.12039184570312,
        "std": 65.61408233642578,
        "min": 20.015625,
        "max": 800.0,
        "25%": 44.40625,
        "75%": 109.0625,
    },
    "bjet_1_eta": {
        "mean": -0.0003414266975596547,
        "std": 1.1415698528289795,
        "min": -2.5,
        "max": 2.5,
        "25%": -0.85595703125,
        "75%": 0.85400390625,
    },
    "jetfwd_n": {
        "mean": 0.5325415730476379,
        "std": 0.7571367025375366,
        "min": 0.0,
        "max": 8.0,
        "25%": 0.0,
        "75%": 1.0,
    },
    "ljet_n": {
        "mean": 1.246351957321167,
        "std": 1.238638162612915,
        "min": 0.0,
        "max": 12.0,
        "25%": 0.0,
        "75%": 2.0,
    },
    "ljet_1_pt": {
        "mean": 52.295406341552734,
        "std": 72.288330078125,
        "min": 0.0,
        "max": 1000.0,
        "25%": 0.0,
        "75%": 69.9375,
    },
    "ljet_1_eta": {
        "mean": -0.9847895503044128,
        "std": 1.7678325176239014,
        "min": -3.0,
        "max": 2.39990234375,
        "25%": -3.0,
        "75%": 0.542236328125,
    },
    "deltar_bjet1_dimuon": {
        "mean": 2.731287717819214,
        "std": 1.0176572799682617,
        "min": 0.0019766842015087605,
        "max": 7.5,
        "25%": 2.1013794541358948,
        "75%": 3.2591171860694885,
    },
    "deltapt_bjet1_dimuon": {
        "mean": 47.54671859741211,
        "std": 50.70753860473633,
        "min": 1.8801783880917355e-05,
        "max": 700.0,
        "25%": 14.579093217849731,
        "75%": 63.64512348175049,
    },
    "deltaeta_bjet1_dimuon": {
        "mean": 1.41896390914917,
        "std": 1.1353912353515625,
        "min": 1.3952715107734548e-06,
        "max": 7.0,
        "25%": 0.5399923771619797,
        "75%": 2.0178425908088684,
    },
    "deltaphi_bjet1_dimuon": {
        "mean": 2.1066300868988037,
        "std": 0.8685312271118164,
        "min": 6.456276878452627e-06,
        "max": 3.141592502593994,
        "25%": 1.5165218114852905,
        "75%": 2.842975437641144,
    }}


def retrieve_stat(stats: dict, which: str, columns: list) -> np.ndarray:
    assert stats == STATS_CAT1

    return np.array([stats[col][which] for col in columns])


def retrieve_clip(ranges: dict,  columns: list) -> np.ndarray:
    assert ranges == CLIP_CAT1

    return np.array([ranges[col] for col in columns])


def get_test_from_dataset(dataset: Dataset, process: str, tanb: float = None, mass=None, interval=None):
    s = dataset.signal
    b = dataset.background
    
    if isinstance(mass, (float, int)):
        s = s[s['mA'] == mass]
        
    if isinstance(interval, (list, tuple, np.ndarray)):
        b = b[(b['dimuon_mass'] > interval[0]) & (b['dimuon_mass'] < interval[1])]
    
    b = b[b['name'] != 'ZMM'].copy()
    
    assert process in s['process'].unique()
    mask = s['process'] == process

    if isinstance(tanb, (int, float)):
        assert tanb in s['tanbeta'].unique()
        mask = mask & (s['tanbeta'] == tanb)
    
    s = s[mask].copy()
    
    ds = Dataset()
    ds.load(signal=s, bkg=b, feature_columns=dataset.columns['feature'])
    
    if (mass is None) and (interval is None):
        ds.mass_intervals = dataset.mass_intervals

    return ds


class CutBased:
    """Classification rules of the CMS cut-based analisys"""
    def __init__(self, bjet_n=-2, ljet_n=-1, met_pt=3, bkg_label=-1.0):
        self.cut_bjet_n = lambda x: x[:, bjet_n] == 1.0
        self.cut_ljet_n = lambda x: x[:, ljet_n] < 2.0
        self.cut_met_pt = lambda x: x[:, met_pt] < 40.0
        
        self.bkg_label = float(bkg_label)
        
    def predict(self, x, *args, **kwargs):
        if isinstance(x, dict):
            x = x['x']
        
        y = tf.logical_and(self.cut_bjet_n(x), self.cut_ljet_n(x))
        y = tf.logical_and(y, self.cut_met_pt(x))
        
        y = tf.cast(y, dtype=tf.float32)
        y = tf.where(y == 0.0, self.bkg_label, 1.0)
        
        return tf.reshape(y, shape=(-1, 1))

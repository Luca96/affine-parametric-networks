import numpy as np
import pandas as pd
import tensorflow as tf

from script import utils
from script.datasets import Dataset


# constants
CLIP_CAT1 = {'dimuon_deltar': (0.5, 5.5), 'dimuon_deltaphi': (-np.inf, np.inf), 
             'dimuon_deltaeta': (-np.inf, np.inf), 'dimuon_pt': (-np.inf, np.inf), 
             'met_pt': (0.0, 500), 'met_phi': (-np.inf, np.inf), 'bjet_n': (0.0, 4.0), 
             'bjet_1_pt': (0.0, 800), 'bjet_1_eta': (-np.inf, np.inf), 
             'jetfwd_n': (0.0, 5.0), 'ljet_n': (0.0, 8.0), 'ljet_1_pt': (0.0, 1000), 
             'ljet_1_eta': (-np.inf, np.inf), 'deltar_bjet1_dimuon': (0.0, 7.5), 
             'deltapt_bjet1_dimuon': (0.0, 650), 'deltaeta_bjet1_dimuon': (0.0, 7), 
             'deltaphi_bjet1_dimuon': (-np.inf, np.inf)}


CLIP_CAT2 = {'dimuon_deltar': (0.35, 5.55), 'met_pt': (0.0, 650), 'ljet_1_pt': (0.0, 1000), 
             'ljet_n': (0.0, 8.0), 'dimuon_pt': (0.0, 1250), 'jetfwd_n': (0.0, 6.0), 
             'dimuon_deltaphi': (-np.inf, np.inf), 'dimuon_deltaeta': (-np.inf, np.inf), 
             'ljet_1_eta': (-np.inf, np.inf), 'met_phi': (-np.inf, np.inf)}


STATS_CAT1 = {
    "dimuon_deltar": {
        "mean": 2.874802350997925,
        "std": 0.584618091583252,
        "min": 0.5,
        "max": 5.5,
        "25%": 2.510730028152466,
        "75%": 3.2035751342773438,
    },
    "dimuon_deltaphi": {
        "mean": 2.3927478790283203,
        "std": 0.6752430200576782,
        "min": 0.0,
        "max": 3.141587495803833,
        "25%": 2.06284236907959,
        "75%": 2.9164860248565674,
    },
    "dimuon_deltaeta": {
        "mean": 1.293296456336975,
        "std": 0.8674412369728088,
        "min": 0.0,
        "max": 4.7802734375,
        "25%": 0.57904052734375,
        "75%": 1.878173828125,
    },
    "dimuon_pt": {
        "mean": 86.30879211425781,
        "std": 72.81925964355469,
        "min": 0.043848637491464615,
        "max": 2360.044189453125,
        "25%": 43.86759662628174,
        "75%": 105.61159706115723,
    },
    "met_pt": {
        "mean": 63.946590423583984,
        "std": 46.89484405517578,
        "min": 0.026056507602334023,
        "max": 500.0,
        "25%": 30.163416385650635,
        "75%": 85.9972095489502,
    },
    "met_phi": {
        "mean": 0.017355069518089294,
        "std": 1.8695093393325806,
        "min": -3.1416015625,
        "max": 3.1416015625,
        "25%": -1.65478515625,
        "75%": 1.68798828125,
    },
    "bjet_n": {
        "mean": 1.3534225225448608,
        "std": 0.5063077807426453,
        "min": 1.0,
        "max": 4.0,
        "25%": 1.0,
        "75%": 2.0,
    },
    "bjet_1_pt": {
        "mean": 87.1014175415039,
        "std": 65.55415344238281,
        "min": 20.015625,
        "max": 800.0,
        "25%": 44.40625,
        "75%": 109.0625,
    },
    "bjet_1_eta": {
        "mean": 0.0009099641465581954,
        "std": 1.1425267457962036,
        "min": -2.5,
        "max": 2.5,
        "25%": -0.855224609375,
        "75%": 0.8564453125,
    },
    "jetfwd_n": {
        "mean": 0.5330425500869751,
        "std": 0.7574427127838135,
        "min": 0.0,
        "max": 5.0,
        "25%": 0.0,
        "75%": 1.0,
    },
    "ljet_n": {
        "mean": 1.2462241649627686,
        "std": 1.237859845161438,
        "min": 0.0,
        "max": 8.0,
        "25%": 0.0,
        "75%": 2.0,
    },
    "ljet_1_pt": {
        "mean": 52.347652435302734,
        "std": 72.39195251464844,
        "min": 0.0,
        "max": 1000.0,
        "25%": 0.0,
        "75%": 69.9375,
    },
    "ljet_1_eta": {
        "mean": -0.9847497940063477,
        "std": 1.7673366069793701,
        "min": -3.0,
        "max": 2.39990234375,
        "25%": -3.0,
        "75%": 0.542236328125,
    },
    "deltar_bjet1_dimuon": {
        "mean": 2.7309765815734863,
        "std": 1.0174551010131836,
        "min": 0.0019766842015087605,
        "max": 7.5,
        "25%": 2.099921405315399,
        "75%": 3.258849322795868,
    },
    "deltapt_bjet1_dimuon": {
        "mean": 47.54897689819336,
        "std": 50.52833557128906,
        "min": 5.8939276641467586e-05,
        "max": 650.0,
        "25%": 14.613624811172485,
        "75%": 63.627346992492676,
    },
    "deltaeta_bjet1_dimuon": {
        "mean": 1.4190870523452759,
        "std": 1.1352860927581787,
        "min": 1.3952715107734548e-06,
        "max": 7.0,
        "25%": 0.5398689061403275,
        "75%": 2.017880380153656,
    },
    "deltaphi_bjet1_dimuon": {
        "mean": 2.1060051918029785,
        "std": 0.8687716722488403,
        "min": 7.735256076557562e-06,
        "max": 3.141592502593994,
        "25%": 1.5140483677387238,
        "75%": 2.8429946899414062,
    }}


STATS_CAT2 = {
    "dimuon_deltar": {
        "mean": 3.052746534347534,
        "std": 0.5257927775382996,
        "min": 0.3499999940395355,
        "max": 5.550000190734863,
        "25%": 2.855406105518341,
        "75%": 3.2983036041259766,
    },
    "dimuon_deltaphi": {
        "mean": 2.745892286300659,
        "std": 0.5254919528961182,
        "min": 0.0,
        "max": 3.1415913105010986,
        "25%": 2.6296207904815674,
        "75%": 3.080047607421875,
    },
    "dimuon_deltaeta": {
        "mean": 1.0758029222488403,
        "std": 0.7888643145561218,
        "min": 0.0,
        "max": 4.7841796875,
        "25%": 0.437469482421875,
        "75%": 1.565673828125,
    },
    "met_pt": {
        "mean": 35.02534484863281,
        "std": 32.16143035888672,
        "min": 0.014294489286839962,
        "max": 650.0,
        "25%": 16.30089282989502,
        "75%": 42.52799987792969,
    },
    "ljet_1_pt": {
        "mean": 31.83565330505371,
        "std": 57.15574264526367,
        "min": 0.0,
        "max": 1000.0,
        "25%": 0.0,
        "75%": 43.59375,
    },
    "ljet_1_eta": {
        "mean": -1.5924359560012817,
        "std": 1.7281346321105957,
        "min": -3.0,
        "max": 2.39990234375,
        "25%": -3.0,
        "75%": -0.1270751953125,
    },
    "ljet_n": {
        "mean": 0.7642363905906677,
        "std": 1.0397272109985352,
        "min": 0.0,
        "max": 8.0,
        "25%": 0.0,
        "75%": 1.0,
    },
    "dimuon_pt": {
        "mean": 49.441837310791016,
        "std": 60.69640350341797,
        "min": 0.012711829505860806,
        "max": 1250.0,
        "25%": 13.658314943313599,
        "75%": 62.958739280700684,
    },
    "met_phi": {
        "mean": 0.022708848118782043,
        "std": 1.9115512371063232,
        "min": -3.1416015625,
        "max": 3.1416015625,
        "25%": -1.729248046875,
        "75%": 1.7685546875,
    },
    "jetfwd_n": {
        "mean": 0.32691410183906555,
        "std": 0.622191309928894,
        "min": 0.0,
        "max": 6.0,
        "25%": 0.0,
        "75%": 1.0,
    }}


def retrieve_stat(stats: dict, which: str, columns: list) -> np.ndarray:
    assert (stats == STATS_CAT1) or (stats == STATS_CAT2)

    return np.array([stats[col][which] for col in columns])


def retrieve_clip(ranges: dict,  columns: list) -> np.ndarray:
    assert (ranges == CLIP_CAT1) or (ranges == CLIP_CAT2)

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


class IndividualNNs:
    """Wraps a set of networks trained on one mA as a whole"""
    def __init__(self, dataset: Dataset, mapping: dict, **kwargs):
        self.models = {}
        
        # load models
        for mass, model_or_path in mapping.items():
            if isinstance(model_or_path, str):
                path = model_or_path
                
                model = utils.get_compiled_non_parametric(dataset, **kwargs)
                utils.load_from_checkpoint(model, path=path)
            else:
                model = model_or_path
            
            self.models[mass] = model
    
    @classmethod
    def load(cls, dataset: Dataset, path_format: str, **kwargs):
        mapping = {}
        
        for mass in dataset.unique_signal_mass:
            mapping[mass] = path_format.format(int(mass))
        
        return cls(dataset, mapping, **kwargs)
    
    def predict(self, x, **kwargs):
        # use the right `model` according to `x['m']` (i.e. mass)
        m = x['m']
        z = np.empty(m.shape, dtype=np.float32)
        
        for mass, model in self.models.items():
            mask = tf.squeeze(m == mass)
            
            if tf.reduce_sum(tf.cast(mask, dtype=tf.int32)) <= 0:
                # no `mass` in input data
                continue
            
            # predict
            x_m = {k: tf.boolean_mask(v, mask) for k, v in x.items()}
            z_m = model.predict(x_m, **kwargs)
            
            z[mask] = z_m
        
        return z

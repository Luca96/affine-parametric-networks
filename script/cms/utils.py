import numpy as np
import pandas as pd
import tensorflow as tf

from script import utils
from script.datasets import Dataset


def retrieve_stat(stats: dict, which: str, columns: list) -> np.ndarray:
    return np.array([stats[col][which] for col in columns])


def retrieve_clip(ranges: dict,  columns: list) -> np.ndarray:
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

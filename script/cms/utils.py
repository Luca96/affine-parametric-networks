import numpy as np
import pandas as pd
import tensorflow as tf

from script import utils
from script.datasets import Dataset


def retrieve_stat(stats: dict, which: str, columns: list) -> np.ndarray:
    return np.array([stats[col][which] for col in columns])


def retrieve_clip(ranges: dict,  columns: list) -> np.ndarray:
    return np.array([ranges[col] for col in columns])


class IndividualNNs:
    """Wraps a set of networks trained on one mass as a whole"""
    def __init__(self, dataset: Dataset, mapping: dict, **kwargs):
        self.models = {}
        self.should_inspect = kwargs.get('inspect', False)

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

        if not self.should_inspect:
            z = np.empty(m.shape, dtype=np.float32)
        else:
            z = dict(y=np.empty(m.shape, dtype=np.float32),
                     r=None)
        
        for mass, model in self.models.items():
            mask = tf.squeeze(m == mass)

            if tf.reduce_sum(tf.cast(mask, dtype=tf.int32)) <= 0:
                # no `mass` in input data
                continue
            
            # predict
            x_m = {k: tf.boolean_mask(v, mask) for k, v in x.items()}
            z_m = model.predict(x_m, **kwargs)

            if not self.should_inspect:
                if isinstance(z_m, dict):
                    z[mask] = z_m['y']
                else:
                    z[mask] = z_m
            else:
                z['y'][mask] = np.squeeze(z_m['y'])

                if z['r'] is None:
                    shape = (m.shape[0], z_m['r'].shape[-1])
                    z['r'] = np.empty(shape, dtype=np.float32)

                z['r'][mask] = z_m['r']
        
        return z

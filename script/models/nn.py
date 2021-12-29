"""Non-parametric Neural Network"""

import tensorflow as tf

from tensorflow.keras.layers import *

from script import utils
from script.models.pnn import PNN

from typing import Dict


class NN(PNN):
    """A NON-parametric Neural Network model; still has a mass input but it's NOT used"""
    
    def __init__(self, *args, name='NonParam-NN', **kwargs):
    	super().__init__(*args, name=name, **kwargs)

    def structure(self, shapes: dict, units=[128, 128], activation='relu', dropout=0.0, linear=False,
                  preprocess: Dict[str, list] = None, **kwargs) -> tuple:
        assert len(units) > 1

        output_args = kwargs.pop('output', {})
        weight_init = kwargs.pop('kernel_initializer', 'glorot_uniform')

        if activation == 'selu':
            is_selu = True
            kwargs['kernel_initializer'] = tf.initializers.lecun_normal(seed=utils.SEED)
        else:
            is_selu = False
            kwargs['kernel_initializer'] = weight_init

        apply_dropout = dropout > 0.0
        
        inputs = self.inputs_from_shapes(shapes)
        preproc_inp = self.apply_preprocessing(inputs, preprocess)
        x = preproc_inp['x']
        
        if linear:
            x = Linear(units=units.pop(0), **kwargs)(x)

        for num_units in units:
            x = Dense(units=num_units, activation=activation, **kwargs)(x)
            
            if apply_dropout:
                if is_selu:
                    x = AlphaDropout(rate=dropout, seed=utils.SEED)(x)
                else:
                    x = Dropout(rate=dropout, seed=utils.SEED)(x)
        
        if is_selu:
            kwargs['kernel_initializer'] = weight_init

        return inputs, self.output_layer(layer=x, **output_args)

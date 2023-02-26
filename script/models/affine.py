import tensorflow as tf

from tensorflow.keras.layers import *

from script.models import PNN
from script.models.layers import AffineConditioning

from typing import Dict


class AffinePNN(PNN):
    """A PNN that uses affine-conditioning on all layers"""
        
    def structure(self, shapes: dict, activation='relu', dropout=0.0, feature_noise=0.0, mass_noise=0.0, 
                  embed=None, affine={}, batch_normalization=False, shared=False, 
                  preprocess: Dict[str, list] = None, **kwargs) -> tuple:
        inputs = self.inputs_from_shapes(shapes)
        preproc_inp = self.apply_preprocessing(inputs, preprocess)

        apply_dropout = dropout > 0.0
        output_args = kwargs.pop('output', {})
        units = kwargs.pop('units')
        
        if activation == 'leaky_relu':
            activation = tf.nn.leaky_relu
        
        x = preproc_inp['x']
        m = preproc_inp['m']

        if batch_normalization:
            x = BatchNormalization()(x)
        
        if feature_noise > 0.0:
            x = GaussianNoise(stddev=feature_noise)(x)
        
        for i, unit in enumerate(units):
            x = Dense(units=unit, activation=activation, **kwargs)(x)

            if mass_noise > 0.0:
                noisy_m = GaussianNoise(stddev=mass_noise)(m)
                x = AffineConditioning(name=f'affine-{i}', **affine)([x, noisy_m])
            else:
                x = AffineConditioning(name=f'affine-{i}', **affine)([x, m])

            if batch_normalization:
                x = BatchNormalization()(x)

            if apply_dropout:
                x = Dropout(rate=dropout)(x)
        
        return inputs, self.output_layer(layer=x, **output_args)

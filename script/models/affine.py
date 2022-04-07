import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *

from script import utils
from script.utils import assert_2d_array
from script.models import PNN
from script.models.layers import AffineConditioning, MassEmbeddingLayer

from typing import Union, List, Dict


class AffinePNN(PNN):
    """A PNN that uses affine-conditioning for all layers"""
        
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

        if kwargs.pop('concat', False):
            x = concatenate([x, m])

        if batch_normalization:
            x = BatchNormalization()(x)
        
        if feature_noise > 0.0:
            x = GaussianNoise(stddev=feature_noise)(x)
        
        if isinstance(embed, dict):
            m = MassEmbeddingLayer(**embed)(m)

        if shared:
            # assert all units are the same
            assert len(set(units)) == 1, 'all `units` must be the same when `shared=True`'
            
            affine = AffineConditioning(name='affine', **affine)
            
            for i, unit in enumerate(units):
                x = Dense(units=unit, activation=activation, **kwargs)(x)
                x = affine([x, m])  # the affine-conditioning layer is shared across the whole architecture
        else:
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
    
    @tf.function
    def train_step(self, batch):
        if isinstance(batch, tuple) and len(batch) == 1:
            batch = batch[0]

        if len(batch) == 3:
            x, labels, sample_weight = batch
        else:
            x, labels = batch
        
            sample_weight = tf.ones_like(labels)
            sample_weight = tf.reduce_mean(sample_weight, axis=-1)[:, tf.newaxis]  # handles multi-class labels
        
        with tf.GradientTape() as tape:
            classes = self(x, training=True)
            
            loss = self.compiled_loss(labels, classes, 
                                      regularization_losses=self.losses, sample_weight=sample_weight)
            total_loss = loss
        
        weight_norm, global_norm, lr = self.apply_gradients(tape, total_loss)
        self.lr.on_step()
        
        debug = self.update_metrics(labels, classes, sample_weight=sample_weight)
        debug['loss'] = tf.reduce_mean(total_loss)
        debug['class-loss'] = tf.reduce_mean(loss)
        debug['lr'] = lr
        debug['grad-norm'] = global_norm
        debug['weight-norm'] = weight_norm
        debug['reg-losses'] = tf.reduce_sum(self.losses)

        return debug

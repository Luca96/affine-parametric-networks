import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *

from script import utils
from script.utils import assert_2d_array, tf_jensen_shannon_divergence
from script.models import PNN
from script.models.layers import AffineConditioning, MassEmbeddingLayer

from typing import Union, List


class AffinePNN(PNN):
    """A PNN that uses affine-conditioning for all layers"""
    
    def __init__(self, *args, mass_weights=None, mass_scaler=None, mass_values=None, adversarial=0.0,
                 mass_intervals: Union[np.ndarray, List[tuple]] = None, track_mass_reliance=False, **kwargs):
        super().__init__(*args, mass_values=mass_values, adversarial=adversarial, track_mass_reliance=track_mass_reliance, **kwargs)
        
        # mass weights       
        if isinstance(mass_weights, (list, tuple, np.ndarray)):
            assert len(mass_weights) == len(mass_intervals)
            
            intervals = np.asarray(mass_intervals)
            assert_2d_array(intervals)
            
            self.mass_low = intervals[:, 0]
            self.mass_high = intervals[:, 1]
            
            # scale bins, since input mass is scaled
            if mass_scaler is not None:
                self.mass_low = mass_scaler.transform(np.reshape(self.mass_low, newshape=(-1, 1)))
                self.mass_low = tf.squeeze(self.mass_low)
                
                self.mass_high = mass_scaler.transform(np.reshape(self.mass_high, newshape=(-1, 1)))
                self.mass_high = tf.squeeze(self.mass_high)
            
            self.mass_low = tf.cast(self.mass_low, dtype=tf.float32)
            self.mass_high = tf.cast(self.mass_high, dtype=tf.float32)
            
            self.mass_weights = tf.squeeze(mass_weights)
        else:
            self.mass_weights = None
        
    def structure(self, shapes: dict, activation='relu', dropout=0.0, feature_noise=0.0, mass_noise=0.0, 
                  embed=None, affine={}, batch_normalization=False, shared=False, **kwargs) -> tuple:
        inputs = self.inputs_from_shapes(shapes)
        
        apply_dropout = dropout > 0.0
        output_args = kwargs.pop('output', {})
        units = kwargs.pop('units')
        
        if activation == 'leaky_relu':
            activation = tf.nn.leaky_relu
        
        x = inputs['x']
        m = inputs['m']

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
        
            sample_weight = self.get_mass_weights(features=x, labels=labels)
        
        # sample background mass
        if self.sample_mass_for_bkg:
            x = self.sample_mass(x, labels)

        if self.should_be_adversarial:
            fake_mass = self.get_adversarial_mass(mass_batch=x['m'])
        
        if self.should_fool:
            x_fool, y_fool = self.get_fooling_batch(x, labels)

        with tf.GradientTape() as tape:
            classes = self(x, training=True)
            
            loss = self.compiled_loss(labels, classes, 
                                      regularization_losses=self.losses, sample_weight=sample_weight)
            
            if self.should_fool:
                fool_loss = self.compiled_loss(y_fool, self(x_fool, training=True))
                fool_loss = tf.reduce_mean(fool_loss) * self.fool_coeff
            else:
                fool_loss = 0.0

            if self.should_be_adversarial:
                adv_loss = self.adversarial_loss(features=x['x'], prob=classes, adversarial_mass=fake_mass)
                adv_loss *= self.adversarial_coeff
            else:
                adv_loss = 0.0
            
            total_loss = loss + adv_loss + fool_loss
        
        weight_norm, global_norm, lr = self.apply_gradients(tape, total_loss)
        self.lr.on_step()
        
        debug = self.update_metrics(labels, classes, sample_weight=sample_weight)
        debug['loss'] = tf.reduce_mean(total_loss)
        debug['fool-loss'] = fool_loss
        debug['class-loss'] = tf.reduce_mean(loss)
        debug['lr'] = lr
        debug['grad-norm'] = global_norm
        debug['weight-norm'] = weight_norm
        debug['reg-losses'] = tf.reduce_sum(self.losses)
        debug['adversarial-loss'] = adv_loss

        if self.should_track_mass_reliance:
            debug['mass-reliance'] = self.get_mass_reliance(x, true=labels)
        
        return debug
    
    def get_mass_weights(self, features, labels):
        if ('m' in features) and tf.is_tensor(self.mass_weights):
            mass = tf.cast(features['m'], dtype=tf.float32)
            
            # find which mass falls in which bin
            mask = (mass >= self.mass_low) & (mass < self.mass_high)
            bins = tf.argmax(tf.cast(mask, dtype=tf.int32), axis=-1)

            # index by bin and label
            indices = tf.concat([bins[:, None], tf.cast(labels, dtype=tf.int64)], axis=-1)

            # retrieve weights
            mass_weights = tf.gather_nd(self.mass_weights, indices)
            mass_weights = tf.cast(mass_weights, dtype=tf.float32)[:, None]
        else:
            mass_weights = tf.ones_like(labels)
            mass_weights = tf.reduce_mean(mass_weights, axis=-1)[:, None]  # handles multi-class labels
            
        return mass_weights


class DcPNN(AffinePNN): 
    """Double-Conditioned PNN: concatenation conditioning is combined with affine-conditioning"""
    
    def __init__(self, *args, name='DC-pNN', **kwargs):
        super().__init__(*args, name=name, **kwargs)
    
    def structure(self, shapes: dict, activation='relu', affine={}, **kwargs) -> tuple:
        inputs = self.inputs_from_shapes(shapes)
        
        units = kwargs.pop('units')
        output_args = kwargs.pop('output', {})
        
        x = inputs['x']
        m = inputs['m']
        
        x = concatenate([x, m], name='concat')
        
        for i, units in enumerate(units):
            # dense -> affine -> concat
            x = Dense(units=units, activation=activation, name=f'dense-{i}', **kwargs)(x)
            
            x = AffineConditioning(name=f'affine-{i}', **affine)([x, m])
            x = concatenate([x, m], name=f'concat-{i}')

        return inputs, self.output_layer(layer=x, **output_args)

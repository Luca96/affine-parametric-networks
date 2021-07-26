"""Implementation of Parametric NNs as detailed by Baldi et al. 2016"""

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from script.utils import tf_global_norm
from script.models.parameters import DynamicParameter

from typing import Dict


class PNN(tf.keras.Model):
    """A Parametric Neural Network (PNN) model"""
    
    def __init__(self, input_shapes: dict, num_classes=2, **kwargs):
        assert num_classes >= 2

        name = kwargs.pop('name', 'ParametricNN')
        self.num_classes = int(num_classes)
        
        inputs, outputs = self.structure(input_shapes, **kwargs)
        super().__init__(inputs, outputs, name=name)

        self.lr = None
        
    def compile(self, optimizer_class=Adam, loss='binary_crossentropy', metrics=None, lr=0.001, **kwargs):
        self.lr = DynamicParameter.create(value=lr)
        optimizer = optimizer_class(learning_rate=self.lr, **kwargs)
        
        if self.num_classes > 2:
            loss = 'categorical_crossentropy'
        
        super().compile(optimizer, loss, metrics)

    def structure(self, shapes: dict, units=[128, 128], activation='relu', dropout=0.0, **kwargs) -> tuple:
        inspect = kwargs.pop('inspect', False)
        apply_dropout = dropout > 0.0
        
        inputs = self.inputs_from_shapes(shapes)
        x = concatenate(list(inputs.values()))
            
        for num_units in units:
            x = Dense(units=num_units, activation=activation, **kwargs)(x)
            
            if apply_dropout:
                x = Dropout(rate=dropout)(x)
        
        if self.num_classes == 2:
            # binary classification
            outputs = Dense(units=1, activation='sigmoid', name='classes', **kwargs)(x)
        else:
            # multi-class classification
            outputs = Dense(units=self.num_classes, activation='softmax', name='classes', **kwargs)(x)

        if inspect:
            return inputs, [outputs, x]

        return inputs, outputs
    
    @tf.function
    def train_step(self, batch):
        if isinstance(batch, tuple) and len(batch) == 1:
            batch = batch[0]
        
        if len(batch) == 3:
            x, labels, sample_weight = batch
        else:
            x, labels = batch
            
            sample_weight = tf.ones_like(labels)
            sample_weight = tf.reduce_mean(sample_weight, axis=-1)[:, None]  # handles multi-class labels
        
        with tf.GradientTape() as tape:
            classes = self(x, training=True)
            
            loss = self.compiled_loss(labels, classes, 
                                      regularization_losses=self.losses, sample_weight=sample_weight)
            
        weight_norm, global_norm, lr = self.apply_gradients(tape, loss)
        self.lr.on_step()

        debug = self.update_metrics(labels, classes, sample_weight=sample_weight)
        debug['loss'] = tf.reduce_mean(loss)
        debug['lr'] = lr
        debug['norm'] = global_norm
        debug['weight-norm'] = weight_norm
        debug['reg-losses'] = tf.reduce_sum(self.losses)
        
        return debug
    
    def apply_gradients(self, tape, loss):
        variables = self.trainable_variables

        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return tf_global_norm(variables), tf_global_norm(grads), self.lr.value
    
    def update_metrics(self, true, predicted, sample_weight=None) -> dict:
        self.compiled_metrics.update_state(true, predicted, sample_weight=sample_weight)

        return {metric.name: metric.result() for metric in self.metrics}

    @staticmethod
    def inputs_from_shapes(shapes: Dict[str, tuple]) -> Dict[str, Input]:
        return {name: Input(shape=shape, name=name) for name, shape in shapes.items()}

"""Implementation of Parametric NNs"""

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras import optimizers as tfo

from script import utils
from script.utils import tf_global_norm
from script.models.parameters import DynamicParameter
from script.models import layers as my_layers

from typing import Dict


class PNN(tf.keras.Model):
    """A Parametric Neural Network (PNN) model"""

    def __init__(self, input_shapes: dict, num_classes=2, lambda_=1.0, **kwargs):
        assert num_classes >= 2

        name = kwargs.pop('name', 'ParametricNN')
        self.num_classes = int(num_classes)

        inputs, outputs = self.structure(input_shapes, **kwargs)
        super().__init__(inputs, outputs, name=name)

        self.lr = None
        self.lambda_ = tf.reshape(tf.constant(lambda_, dtype=tf.float32), shape=[1])

    def compile(self, optimizer_class=tfo.Adam, loss=None, metrics=None, lr=0.001, **kwargs):
        self.lr = DynamicParameter.create(value=lr)

        if isinstance(optimizer_class, str):
            optimizer_class = dict(sgd=tfo.SGD, adam=tfo.Adam, nadam=tfo.Nadam,
                                   rmsprop=tfo.RMSprop)[optimizer_class.lower()]

        optimizer = optimizer_class(learning_rate=self.lr, **kwargs)

        if loss is None:
            loss = 'categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'

        super().compile(optimizer, loss, metrics)

    def structure(self, shapes: dict, units=[128, 128], activation='relu', dropout=0.0, linear=False,
                  noise=0.0, preprocess: Dict[str, list] = None, batch_normalization=False,
                  conditioning: dict = None, **kwargs) -> tuple:
        assert len(units) > 1

        inspect = kwargs.pop('inspect', False)
        output_args = kwargs.pop('output', {})
        weight_init = kwargs.pop('kernel_initializer', 'glorot_uniform')

        conditioning = conditioning or {}
        conditioning.setdefault('method', 'concat')
        conditioning.setdefault('place', 'start')
        assert conditioning['place'] in ['start', 'all', 'end']

        if activation == 'selu':
            is_selu = True
            # TODO: use variance scaling, instead
            kwargs['kernel_initializer'] = tf.initializers.lecun_normal(seed=utils.SEED)
        else:
            is_selu = False
            kwargs['kernel_initializer'] = weight_init

            if activation == 'leaky_relu':
                activation = tf.nn.leaky_relu

        apply_dropout = dropout > 0.0

        inputs = self.inputs_from_shapes(shapes)
        preproc_inp = self.apply_preprocessing(inputs, preprocess)

        if noise > 0.0:
            m = preproc_inp['m']
            m = GaussianNoise(stddev=noise)(m)
        else:
            m = preproc_inp['m']

        if conditioning['place'] == 'start':
            x = self._get_conditioning(method=conditioning['method'])([preproc_inp['x'], m])
        else:
            x = preproc_inp['x']

        if linear:
            x = my_layers.Linear(units=units.pop(0), **kwargs)(x)

        for num_units in units:
            x = Dense(units=num_units, activation=activation, **kwargs)(x)

            if conditioning['place'] == 'all':
                x = self._get_conditioning(method=conditioning['method'])([x, m])

            if batch_normalization:
                x = BatchNormalization()(x)

            if apply_dropout:
                if is_selu:
                    x = AlphaDropout(rate=dropout, seed=utils.SEED)(x)
                else:
                    x = Dropout(rate=dropout, seed=utils.SEED)(x)

        if is_selu:
            kwargs['kernel_initializer'] = weight_init

        if conditioning['place'] == 'end':
            x = self._get_conditioning(method=conditioning['method'])([x, m])

        outputs = self.output_layer(layer=x, **output_args)

        if inspect:
            return inputs, [outputs, x]

        return inputs, outputs

    def output_layer(self, layer: Layer, name='classes', **kwargs):
        if self.num_classes == 2:
            return Dense(units=1, activation=kwargs.pop('activation', 'sigmoid'), name=name,
                         **kwargs)(layer)

        # multi-class classification
        return Dense(units=self.num_classes, activation=kwargs.pop('activation', 'softmax'),
                     name=name, **kwargs)(layer)

    def apply_preprocessing(self, inputs: dict, preprocess: Dict[str, list] = None) -> dict:
        if preprocess is None:
            return inputs

        inputs = {k: v for k, v in inputs.items()}  # make a copy

        for k, layers in preprocess.items():
            if k not in inputs:
                continue

            in_layer = inputs[k]

            for layer in layers:
                in_layer = layer(in_layer)

            inputs[k] = in_layer

        return inputs

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

            total_loss = self.compiled_loss(labels, classes, regularization_losses=self.losses,
                                            sample_weight=sample_weight)

        weight_norm, global_norm, lr = self.apply_gradients(tape, total_loss)
        self.lr.on_step()

        debug = self.update_metrics(labels, classes, sample_weight=sample_weight)
        debug['loss'] = tf.reduce_mean(total_loss)
        debug['reg-losses'] = tf.reduce_sum(self.losses)
        debug['class-loss'] = debug['loss'] - debug['reg-losses']
        debug['lr'] = lr
        debug['grad-norm'] = global_norm
        debug['weight-norm'] = weight_norm
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

    @staticmethod
    def _get_conditioning(method: str) -> Layer:
        method = method.lower()

        if method == 'concat':
            return tf.keras.layers.concatenate

        if method in ['bias', 'biasing']:
            return my_layers.ConditionalBiasing()

        if method in ['scale', 'scaling']:
            return my_layers.ConditionalScaling()

        if method == 'affine':
            return my_layers.AffineConditioning()

        raise ValueError(f'No conditioning method named: "{method}". Try one of [concat, bias, scale, affine]')

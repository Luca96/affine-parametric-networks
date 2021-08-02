"""Definition of custom layers"""

import tensorflow as tf

from tensorflow.keras.layers import *


class Linear(Dense):
    """Linear combination layer"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='linear', **kwargs)


class ConcatConditioning(Layer):
    """Concatenation-based conditioning: first concatenates, then linearly combines"""

    def __init__(self, units: int, name=None, **kwargs):
        super().__init__(name=name)

        self.concat = Concatenate()
        self.dense = Linear(units=int(units), **kwargs)

    def call(self, inputs: list, **kwargs):
        assert isinstance(inputs, (list, tuple))

        x = self.concat(inputs)
        x = self.dense(x)
        return x


class AffineConditioning(Layer):
    """Generalized affine transform-based conditioning layer"""

    def __init__(self, scale_activation='linear', bias_activation='linear', name=None, **kwargs):
        super().__init__(name=name)
        self.kwargs = kwargs
        
        self.scale_activation = scale_activation
        self.bias_activation = bias_activation
        
        self.dense_scale: Dense = None
        self.dense_bias: Dense = None

        self.multiply = Multiply()
        self.add = Add()

    def build(self, input_shape):
        shape, _ = input_shape
        
        self.dense_scale = Dense(units=shape[-1], activation=self.scale_activation, **self.kwargs)
        self.dense_bias = Dense(units=shape[-1], activation=self.bias_activation, **self.kwargs)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 2

        # condition input `x` on `z`
        x, z = inputs

        scale = self.dense_scale(z)
        bias = self.dense_bias(z)

        # apply affine transformation, i.e. y = scale(z) * x + bias(z)
        y = self.multiply([x, scale])
        y = self.add([y, bias])
        return y    


class MassEmbeddingLayer(Layer):
    """Represents a single mass value as a dense vector (embedding)"""
    
    def __init__(self, alpha=1.0, units=2, activation='linear', name=None, **kwargs):
        super().__init__(name=name)
        
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.dense = Dense(units=units, activation=activation, **kwargs)
        
    def call(self, inputs):
        embedding = self.dense(inputs)
        
        self.add_loss(self.alpha * self.pairwise_cosine_similarity(embedding))
        return embedding
        
    def pairwise_cosine_similarity(self, x):
        repeated_x = x[:, None] * tf.ones_like(x)  # entire `x` is repeated `batch_size` times
        tiled_x = x[None, :] * tf.ones_like(x)[:, None]  # each element of `x` is repeated `batch_size` times

        pairwise_sim = tf.keras.losses.cosine_similarity(repeated_x, tiled_x)
        
        loss = 1.0 - tf.math.abs(pairwise_sim)
        loss = tf.reduce_max(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        return -loss


class MassLossLayer(Layer):
    """Adds a MAE loss w.r.t. the input and predicted mass"""

    def __init__(self, alpha: tf.Tensor, loss='mae', **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
        if loss.lower() == 'mae':
            self.loss_fn = lambda x: tf.math.abs(x)
        else:
            self.loss_fn = lambda x: tf.square(x)

    def call(self, inputs, training=False, **kwargs):
        labels, pred_mass, true_mass = inputs
        
        if training:
            loss = self.alpha * self.loss_fn(true_mass - pred_mass)
            self.add_loss(tf.reduce_mean(loss))

        return labels


class DropFeatures(Layer):
    """A layer that randomly drops some columns (i.e. features) of the input tensor"""
    
    def __init__(self, fraction: float, seed=None, **kwargs):
        assert 0.0 < fraction < 1.0
        super().__init__(**kwargs)

        self.fraction = fraction
        self.amount = None
        self.indices = None
        self.seed = seed

    def build(self, input_shape):
        self.amount = int(self.fraction * input_shape[-1])
        self.indices = tf.range(input_shape[-1])  

    def call(self, inputs):
        # select `amount` indices
        indices = tf.random.shuffle(self.indices, seed=self.seed)[:self.amount]

        # build mask
        # https://stackoverflow.com/questions/50152743/mask-tensor-with-0-according-to-value-at-some-specific-index-at-the-inner-most-d
        # https://towardsdatascience.com/how-to-replace-values-by-index-in-a-tensor-with-tensorflow-2-0-510994fe6c5f
        
        # output = mask * inputs
        pass

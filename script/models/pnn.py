"""Implementation of Parametric NNs as detailed by Baldi et al. 2016"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *
from tensorflow.keras import optimizers as tfo

from script import utils
from script.utils import tf_global_norm
from script.models.parameters import DynamicParameter
from script.models.layers import Linear

from typing import Dict, Union


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        
        self.gamma = tf.constant(float(gamma), dtype=tf.float32)
        self.eps = tf.constant(float(eps), dtype=tf.float32)
    
    @tf.function
    def __call__(self, y, p, sample_weight=None):
        # `y = y_true`, `p = y_pred`

        # For numerical stability (so we don't inadvertently take the log of 0)
        p = tf.math.maximum(p, self.eps)
        p = tf.reshape(p, shape=(-1, 1))
        # q = tf.math.maximum(1.0 - p, self.eps)

        # prob of true class
        pt = tf.where(y == 1.0, p, 1.0 - p)

        # Loss for the positive examples
        if tf.is_tensor(sample_weight) or isinstance(sample_weight, (tf.Tensor, np.ndarray)):
            alpha = tf.reshape(sample_weight, shape=(-1, 1))
        else:
            alpha = tf.ones_like(pt)

        # Loss for the negative examples
        loss = -alpha * (1.0 - pt) ** self.gamma * tf.math.log(pt)
        return tf.reduce_mean(loss)


class WrappedAUC:
    def __init__(self, name='wrapped_auc'):
        self.auc = tf.keras.metrics.AUC(name=name)

    def update_state(self, x, y):
        self.auc.update_state(x, y)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()


class PNN(tf.keras.Model):
    """A Parametric Neural Network (PNN) model"""
    
    def __init__(self, input_shapes: dict, num_classes=2, track_mass_reliance=False,  mass_values=None, adversarial=0.0,
                 mass=None, sample_mass=None, bins=20, eps=1e-7, fooling: dict = None, bayesian=False, lambda_=1.0, **kwargs):
        assert num_classes >= 2

        name = kwargs.pop('name', 'ParametricNN')
        self.num_classes = int(num_classes)
        self.is_bayesian = bool(bayesian)
        
        inputs, outputs = self.structure(input_shapes, **kwargs)
        super().__init__(inputs, outputs, name=name)

        self.lr = None
        self.lambda_ = tf.reshape(tf.constant(lambda_, dtype=tf.float32), shape=[1])

        # mass reliance
        if track_mass_reliance:
            self.should_track_mass_reliance = True
            self.auc = WrappedAUC()
        else:
            self.should_track_mass_reliance = False

        # sample mass for bkg events
        if mass is not None:
            assert isinstance(mass, (int, float, list, tuple, tf.Tensor, np.ndarray))
            
            self.mass = tf.constant(mass, dtype=tf.float32)
            self.mass_idx = tf.range(self.mass.shape[0])
            self.sample_mass_for_bkg = True
        else:
            self.sample_mass_for_bkg = False

        if isinstance(sample_mass, bool):
            self.sample_mass_for_bkg = sample_mass

        # fooling loss
        if isinstance(fooling, dict):
            self.fool_coeff = tf.constant(fooling['coeff'], dtype=tf.float32)
            self.fool_size = fooling['size']

            # self.fool_idx = np.random.choice(self.mass_idx, self.fool_size)
            # self.fool_idx = tf.constant(self.fool_idx)

            self.fool_idx = tf.concat([self.mass_idx] * self.fool_size, axis=0)

            self.should_fool = True
        else:
            self.should_fool = False

        # adversarial loss
        if isinstance(mass_values, (tuple, list, np.ndarray)):
            if adversarial != 0.0:
                self.should_be_adversarial = True
                
                self.unique_mass_values = tf.constant(mass_values, dtype=tf.float32)
                self.unique_mass_values = tf.reshape(self.unique_mass_values, shape=(1, -1))
                
                self.adversarial_coeff = tf.constant(adversarial, dtype=tf.float32)
            else:
                self.should_be_adversarial = False
        else:
            self.should_be_adversarial = False

        # # significance (AMS)
        # self.cuts = tf.linspace(0.0, 1.0 + eps, num=bins)
        # self.init_shape = (tf.shape(self.cuts)[0] - 1,)

        # self.sig = tf.Variable(tf.zeros(self.init_shape), trainable=False)
        # self.bkg = tf.Variable(tf.zeros(self.init_shape), trainable=False)

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
                  noise=0.0, preprocess: Dict[str, list] = None, **kwargs) -> tuple:
        assert len(units) > 1

        inspect = kwargs.pop('inspect', False)
        output_args = kwargs.pop('output', {})
        weight_init = kwargs.pop('kernel_initializer', 'glorot_uniform')

        if activation == 'selu':
            is_selu = True
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

        x = concatenate([preproc_inp['x'], m])

        if linear:
            x = Linear(units=units.pop(0), **kwargs)(x)

        for num_units in units:
            if self.is_bayesian:
                x = tfp.layers.DenseFlipout(units=num_units, activation=activation)(x) 
            else:
                x = Dense(units=num_units, activation=activation, **kwargs)(x)
            
            if apply_dropout:
                if is_selu:
                    x = AlphaDropout(rate=dropout, seed=utils.SEED)(x)
                else:
                    x = Dropout(rate=dropout, seed=utils.SEED)(x)
        
        if is_selu:
            kwargs['kernel_initializer'] = weight_init

        outputs = self.output_layer(layer=x, **output_args)

        if inspect:
            return inputs, [outputs, x]

        return inputs, outputs
    
    def output_layer(self, layer: Layer, name='classes', **kwargs):
        if self.num_classes == 2:
            # binary classification
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
        
        # sample background mass
        if self.sample_mass_for_bkg:
            x = self.sample_mass(x, labels)

        if self.should_be_adversarial:
            fake_mass = self.get_adversarial_mass(mass_batch=x['m'])

        if self.should_fool:
            x_fool, y_fool = self.get_fooling_batch(x, labels)

        with tf.GradientTape() as tape:
            classes = self(x, training=True)
            
            if self.is_bayesian:
                kl_loss = tf.reduce_sum(self.losses) * self.lambda_

                loss = self.compiled_loss(labels, classes, sample_weight=sample_weight)
            else:
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

            if self.is_bayesian:
                total_loss += kl_loss

        weight_norm, global_norm, lr = self.apply_gradients(tape, total_loss)
        self.lr.on_step()

        debug = self.update_metrics(labels, classes, sample_weight=sample_weight)
        debug['loss'] = tf.reduce_mean(total_loss)
        debug['class-loss'] = tf.reduce_mean(loss)
        debug['fool-loss'] = fool_loss
        debug['lr'] = lr
        debug['grad-norm'] = global_norm
        debug['weight-norm'] = weight_norm
        debug['adversarial-loss'] = adv_loss
        debug['reg-losses'] = tf.reduce_sum(self.losses)

        if self.is_bayesian:
            debug['kl-loss'] = kl_loss
        
        if self.should_track_mass_reliance:
            debug['mass-reliance'] = self.get_mass_reliance(x, true=labels)

        return debug
    
    def apply_gradients(self, tape, loss):
        variables = self.trainable_variables

        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return tf_global_norm(variables), tf_global_norm(grads), self.lr.value
    
    def update_metrics(self, true, predicted, sample_weight=None) -> dict:
        self.compiled_metrics.update_state(true, predicted, sample_weight=sample_weight)

        return {metric.name: metric.result() for metric in self.metrics}

    def sample_mass(self, x: Dict[str, tf.Tensor], labels: tf.Tensor):
        sig_mask = tf.reshape(labels == 1.0, shape=[-1])
        bkg_mask = tf.logical_not(sig_mask)
        num = tf.shape(bkg_mask)[0]

        # check if there are no sig or bkg samples in batch
        if num == 0 or num == tf.shape(x['x'])[0]:
            return x

        sig = {k: v[sig_mask] for k, v in x.items()}
        
        # (1) concat mass indices "bkg_mask.shape[0]" times
        idx = tf.concat([self.mass_idx] * num, axis=0)

        # (2) randomize indices
        idx = tf.random.shuffle(idx)[:num]

        # (3) gather random mass values
        mass = tf.gather(self.mass, idx)
        mass = tf.reshape(mass, shape=(-1, 1))

        # (4) recreate the input data
        return dict(x=tf.concat([sig['x'], x['x'][bkg_mask]], axis=0),
                    m=tf.concat([sig['m'], mass], axis=0))

    def adversarial_loss(self, features, prob, adversarial_mass):
        # predict using the same features but "fake" (adversarial) mass values
        fake_prob = self(dict(x=features, m=adversarial_mass), training=True)
        
        # maximize the distance (i.e. diversity) between the two predicted probability distributions
        return -tf.reduce_mean(utils.tf_jensen_shannon_divergence(prob, fake_prob))
    
    def get_adversarial_mass(self, mass_batch):
        # repeats unique mass values along the batch-dimension
        tiled_mass = self.unique_mass_values * tf.ones_like(mass_batch)  # shape: (batch_size, M)
        
        # remove the mass values present in `mass_batch`
        mask = tf.logical_not(tiled_mass == mass_batch)
        mask.set_shape((None, None))

        diverse_mass = tf.reshape(tiled_mass[mask], shape=(-1, tf.shape(self.unique_mass_values)[-1] - 1))
        
        # sample one of them along the batch-axis
        sampled_mass = tf.map_fn(lambda x: tf.random.shuffle(x, seed=utils.SEED)[0], diverse_mass)
        return tf.expand_dims(sampled_mass, axis=-1)

    def get_fooling_batch(self, x: Dict[str, tf.Tensor], labels: tf.Tensor):
        sig_mask = tf.reshape(labels == 1.0, shape=[-1])
        num = tf.shape(sig_mask)[0]

        # check if there are no signal samples in batch
        # if num == 0:
        #     size = self.fool_size
        #     return dict(x=x['x'][:size], m=x['m'][:size]), labels[:size]

        # (1) retrieve signal only
        sig = x['x'][sig_mask]

        # (2) randomize indices
        idx = tf.random.shuffle(self.fool_idx)[:self.fool_size]
        
        # (3) gather random mass values, along with signal features
        mass = tf.gather(self.mass, idx)
        mass = tf.reshape(mass, shape=(-1, 1))
        sig_ = tf.gather(sig, idx)

        # (4) create the batch
        # y = tf.zeros(shape=(self.fool_size, 1), dtype=tf.float32)

        # same mass can be chosen, if so put y_i == 1
        y = tf.cast(x['m'] == mass, dtype=tf.float32)

        return dict(x=sig_, m=mass), y

    @staticmethod
    def inputs_from_shapes(shapes: Dict[str, tuple]) -> Dict[str, Input]:
        return {name: Input(shape=shape, name=name) for name, shape in shapes.items()}

    def get_mass_reliance(self, x: dict, true):
        inputs = dict(x=tf.zeros_like(x['x']), m=x['m'])
        pred = self(inputs, training=False)

        self.auc.update_state(true, pred)
        auc = tf.reduce_mean(self.auc.result())
        self.auc.reset_states()

        return 2 * tf.math.abs(auc - 0.5)

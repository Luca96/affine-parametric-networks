"""Dynamic step-dependent parameters: used as custom learning rate schedules"""

import tensorflow as tf 

from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from typing import Union


class DynamicParameter:
    """Interface for learning rate schedule wrappers as dynamic-parameters"""
    def __init__(self):
        self._value = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)
        self.step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

    @property
    def value(self):
        return self._value.value()

    @property
    def variable(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value.assign(value)

    @staticmethod
    def create(value: Union[float, int, LearningRateSchedule], **kwargs):
        """Converts a floating or LearningRateSchedule `value` into a DynamicParameter object"""
        if isinstance(value, (DynamicParameter, ScheduleWrapper)):
            return value

        if isinstance(value, (float, int)):
            return ConstantParameter(value)

        if isinstance(value, LearningRateSchedule):
            return ScheduleWrapper(schedule=value, **kwargs)

        raise ValueError(f'Parameter "value" should be not {type(value)}.')

    def __call__(self, *args, **kwargs):
        return self.value

    def __sub__(self, other):
        self._value.assign_sub(other)

    def __add__(self, other):
        self._value.assign_add(other)

    def serialize(self) -> dict:
        return dict(step=int(self.step.value()))

    def on_step(self):
        self.step.assign_add(delta=1)

    def load(self, config: dict):
        self.step.assign(value=config.get('step', 0))

    def get_config(self) -> dict:
        return {}


class ScheduleWrapper(LearningRateSchedule, DynamicParameter):
    """A wrapper for built-in tf.keras' learning rate schedules"""
    def __init__(self, schedule: LearningRateSchedule, min_value=1e-7, max_value=None):
        super().__init__()
        self.schedule = schedule
        self.min_value = tf.constant(min_value, dtype=tf.float32)

        if isinstance(max_value, (float, int)):
            self.max_value = tf.constant(max_value, dtype=tf.float32)
        else:
            self.max_value = None

        self._value.assign(value=self.schedule.initial_learning_rate)

    def __call__(self, *args, **kwargs):
        self.value = tf.maximum(self.min_value, self.schedule.__call__(self.step))

        if self.max_value:
            self.value = tf.minimum(self.value, self.max_value)

        return self.value

    def get_config(self) -> dict:
        return self.schedule.get_config()


class ConstantParameter(DynamicParameter):
    """A constant learning rate schedule that wraps a constant float learning rate value"""
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value

    def serialize(self) -> dict:
        return {}


class ExponentialDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, steps: int, rate: float, staircase=False, min_value=0.0, max_value=None):
        super().__init__(schedule=schedules.ExponentialDecay(initial_learning_rate=float(initial_value),
                                                             decay_steps=int(steps), decay_rate=float(rate),
                                                             staircase=bool(staircase)),
                         min_value=min_value, max_value=max_value)


class StepDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, steps: int, rate: float, min_value=1e-7, max_value=None):
        super().__init__(schedule=schedules.ExponentialDecay(initial_learning_rate=float(initial_value),
                                                             decay_steps=int(steps), decay_rate=float(rate),
                                                             staircase=True),
                         min_value=min_value, max_value=max_value)


class LinearDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, end_value: float, steps: int, cycle=False):
        super().__init__(schedule=schedules.PolynomialDecay(initial_learning_rate=float(initial_value),
                                                            decay_steps=int(steps), end_learning_rate=float(end_value),
                                                            power=1.0, cycle=bool(cycle)))


class PolynomialDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, end_value: float, steps: int, power=1.0, cycle=False):
        super().__init__(schedule=schedules.PolynomialDecay(initial_learning_rate=float(initial_value),
                                                            decay_steps=int(steps), end_learning_rate=float(end_value),
                                                            power=power, cycle=bool(cycle)))

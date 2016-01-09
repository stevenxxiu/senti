
import itertools
import logging
import warnings
from contextlib import contextmanager

import theano.tensor as T
from keras.layers.core import *
from keras.models import Graph as _Graph

__all__ = ['log_softmax', 'categorical_crossentropy_exp', 'f1', 'LambdaTest', 'geometric_learning_rates', 'Graph']

##
# Helpers
##


def nan_divide(x, y, replace=0):
    return T.switch(T.eq(y, 0), replace, x / y)


def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))


##
# Objectives
##


def categorical_crossentropy_exp(y_true, y_pred):
    if y_true.ndim == y_pred.ndim:
        return -T.sum(y_true * y_pred, axis=1)
    elif y_true.ndim == y_pred.ndim - 1:
        return -y_pred[T.arange(y_true.shape[0]), y_true]
    else:
        raise TypeError('rank mismatch between coding and true distributions')

##
# Metrics
##


def f1(num_classes, classes):
    # noinspection PyShadowingNames
    def f1(y_true, y_pred, class_mode):
        if class_mode == 'categorical':
            y_pred = T.argmax(y_pred, axis=-1)
            if y_true.ndim == 2:
                y_true = T.argmax(y_true, axis=-1)
            tp_sum = T.extra_ops.bincount(y_true, T.eq(y_true, y_pred), minlength=num_classes)[classes]
            pred_sum = T.extra_ops.bincount(y_pred, minlength=num_classes)[classes]
            true_sum = T.extra_ops.bincount(y_true, minlength=num_classes)[classes]
            precision = nan_divide(tp_sum, pred_sum)
            recall = nan_divide(tp_sum, true_sum)
            f_score = nan_divide(2 * precision * recall, precision + recall)
            return T.mean(f_score)
        elif class_mode == 'binary':
            raise NotImplementedError
        else:
            raise Exception('Invalid class mode:' + str(class_mode))

    return f1


##
# Learning Rates
##


@contextmanager
def learning_rate_yielder(func):
    def decorated(model, *args, **kwargs):
        iter_ = func(*args, **kwargs)

        def schedular(_):
            model.lr.set_value(next(iter_))
            return model.lr.get_value()

        return schedular
    return decorated


@learning_rate_yielder
def geometric_learning_rates(init, ratio=None, repeat=None, n=0):
    learning_rate = init
    for i in range(n):
        yield from itertools.repeat([learning_rate], repeat)
        learning_rate *= ratio
    yield from itertools.repeat([learning_rate])


##
# Layers
##


class LambdaTest(Lambda):
    def get_output(self, train=False):
        func = marshal.loads(self.function)
        func = types.FunctionType(func, globals())
        if hasattr(self, 'previous'):
            res = self.previous.get_output(train)
        else:
            res = self.input
        return res if train else func(res)


##
# Models
##


class Graph(_Graph):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self._prev_name = ''

    def add_input(self, name, input_shape=None, batch_input_shape=None, dtype='float'):
        self._prev_name = name
        return super().add_input(name, input_shape, batch_input_shape, dtype)

    # noinspection PyDefaultArgument,PyShadowingBuiltins
    def add_node(
        self, layer, name=None, input=None, inputs=[], merge_mode='concat', concat_axis=-1, dot_axes=-1,
        create_output=False
    ):
        if name is None:
            name = 'node_{}'.format(len(self.nodes))
        if input is None:
            input = self._prev_name
        if len(inputs) == 1:
            input = inputs[0]
            inputs = []
        self._prev_name = name
        return super().add_node(layer, name, input, inputs, merge_mode, concat_axis, dot_axes, create_output)

    # noinspection PyDefaultArgument,PyShadowingBuiltins
    def add_output(self, name, input=None, inputs=[], merge_mode='concat', concat_axis=-1, dot_axes=-1):
        if input is None:
            input = self._prev_name
        return super().add_output(name, input, inputs, merge_mode, concat_axis, dot_axes)

    def _pre_transform(self, data):
        raise NotImplementedError

    def _gen_batches(self, data):
        for i in itertools.count(0):
            logging.info('training set pass {}'.format(i + 1))
            perm = np.random.permutation(len(next(iter(data.values()))))
            data = {key: [value[i] for i in perm] for key, value in data.items()}
            for j in range(0, len(next(iter(data.values()))), self.batch_size):
                yield self._pre_transform({key: value[j:j + self.batch_size] for key, value in data.items()})

    # noinspection PyDefaultArgument,PyMethodOverriding
    def fit(
        self, data, nb_epoch, samples_per_epoch=None, verbose=1, callbacks=[], validation_data=None, class_weight={},
        nb_worker=1
    ):
        for data_ in (data, validation_data):
            for key, value in data_.items():
                data_[key] = list(value)
        validation_data = self._pre_transform(validation_data)
        if samples_per_epoch is None:
            samples_per_epoch = len(next(iter(data.values())))
        return super().fit_generator(
            self._gen_batches(data), samples_per_epoch, nb_epoch, verbose, callbacks, validation_data, class_weight,
            nb_worker
        )

    def predict_proba(self, data, verbose=1):
        preds = self.predict(data, self.batch_size, verbose)
        for cur_preds in preds:
            if cur_preds.min() < 0 or cur_preds.max() > 1:
                warnings.warn('Network returning invalid probability values.')
        return preds

    def predict_classes(self, data, verbose=1):
        proba = self.predict(data, self.batch_size, verbose)
        return proba.argmax(axis=-1)

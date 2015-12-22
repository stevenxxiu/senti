
import numpy as np
import theano.tensor as T
from lasagne.layers import Layer

__all__ = ['log_softmax', 'categorical_crossentropy_exp', 'KMaxPool1DLayer']


def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))


def categorical_crossentropy_exp(log_predictions, targets, batch_size=None):
    if targets.ndim == log_predictions.ndim:
        return -T.sum(targets * log_predictions, axis=1)
    elif targets.ndim == log_predictions.ndim - 1:
        return -log_predictions[np.arange(batch_size), targets]
    else:
        raise TypeError('rank mismatch between coding and true distributions')


class KMaxPool1DLayer(Layer):
    def __init__(self, incoming, k, **kwargs):
        super().__init__(incoming, **kwargs)
        self.k = k

    def get_output_shape_for(self, input_shape):
        res = list(input_shape)
        res[-1] = self.k
        return tuple(res)

    def get_output_for(self, input_, **kwargs):
        return input_[
            T.arange(input_.shape[0]).dimshuffle(0, 'x', 'x'),
            T.arange(input_.shape[1]).dimshuffle('x', 0, 'x'),
            T.sort(T.argsort(input_, axis=-1)[:, :, -self.k:], axis=-1),
        ]

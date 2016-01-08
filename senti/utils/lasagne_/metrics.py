
import numpy as np
import theano.tensor as T
from lasagne.objectives import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

__all__ = ['acc', 'f1', 'rmse']


def acc(y_true, y_pred, theano=False):
    if theano:
        if y_true.ndim == 2:
            y_true = T.argmax(y_true, axis=-1)
        return T.mean(T.eq(y_true, T.argmax(y_pred, axis=-1)))
    else:
        return accuracy_score(y_true, np.argmax(y_pred, axis=-1))


def f1(average_classes):
    # noinspection PyShadowingNames
    def f1(y_true, y_pred, theano=False):
        if theano:
            raise NotImplementedError
        else:
            return np.mean(precision_recall_fscore_support(y_true, np.argmax(y_pred, axis=-1))[2][average_classes])

    return f1


def rmse(y_true, y_pred, theano=False):
    if theano:
        return T.mean(aggregate(squared_error(y_true, y_pred)))**0.5
    else:
        return np.mean((y_true - y_pred)**2)

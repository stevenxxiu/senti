
import theano.tensor as T

__all__ = ['log_softmax', 'categorical_crossentropy_exp']


def log_softmax(x):
    x_diff = x - x.max(1, keepdims=True)
    return x_diff - T.log(T.sum(T.exp(x_diff), axis=1, keepdims=True))


def categorical_crossentropy_exp(y_true, y_pred):
    if y_true.ndim == y_pred.ndim:
        return -T.sum(y_true * y_pred, axis=1)
    elif y_true.ndim == y_pred.ndim - 1:
        return -y_pred[T.arange(y_true.shape[0]), y_true]
    else:
        raise TypeError('rank mismatch between coding and true distributions')

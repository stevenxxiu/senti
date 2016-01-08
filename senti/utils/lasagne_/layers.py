
import theano.tensor as T
from lasagne.layers import Layer

__all__ = ['KMaxPool1DLayer']


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

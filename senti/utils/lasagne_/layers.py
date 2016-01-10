
import theano.tensor as T
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import Layer

__all__ = ['KMaxPool1DLayer', 'SelfInteractionLayer']


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


class SelfInteractionLayer(Layer):
    def __init__(self, incoming, W=init.Uniform(), nonlinearity=nonlinearities.rectify, **kwargs):
        super().__init__(incoming, **kwargs)
        self.W = self.add_param(W, (self.input_shape[1], self.input_shape[1], self.input_shape[1]), name='W')
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input_, **kwargs):
        W = T.tril(self.W, -1)
        interactions = T.batched_dot(T.dot(input_, W), input_)
        interactions = T.sqrt(T.max(interactions, 1e-6))
        return self.nonlinearity(input_ + interactions)

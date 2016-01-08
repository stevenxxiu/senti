
import lasagne
import theano.tensor as T
from lasagne.nonlinearities import *

from senti.utils.lasagne_ import *

__all__ = ['NNShallow']


class NNShallow(NNBase):
    '''
    Trains the upper layers of an existing neural net.
    '''

    def __init__(self, model, num_train):
        super().__init__(model.batch_size)
        self.model = model
        self.target = T.ivector('target')
        l = model.network
        for _ in range(num_train - 1):
            l = l.input_layer
        self._features = theano.function(model.inputs, lasagne.layers.get_output(l.input_layer, deterministic=True))
        self.inputs = [lasagne.layers.get_output_shape(l.input_layer)]
        l_in = lasagne.layers.InputLayer(self.inputs[0])
        l.input_shape = l_in.shape
        l.input_layer = l_in
        l = model.network
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, lasagne.layers.get_output(l)))
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.metrics = model.metrics
        self.network = l
        self.compile()

    def gen_batch(self, X, y=None):
        return self.model.gen_batch(X, y)

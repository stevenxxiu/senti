
import theano.tensor as T
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.updates import *

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
        self._features = theano.function(model.inputs, get_output(l.input_layer, deterministic=True))
        self.inputs = [get_output_shape(l.input_layer)]
        l_in = InputLayer(self.inputs[0])
        l.input_shape = l_in.shape
        l.input_layer = l_in
        l = model.network
        self.probs = T.exp(get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, get_output(l)))
        params = get_all_params(l, trainable=True)
        self.updates = adadelta(self.loss, params)
        self.metrics = model.metrics
        self.network = l
        self.compile()

    def gen_batch(self, X, y=None):
        return self.model.gen_batch(X, y)

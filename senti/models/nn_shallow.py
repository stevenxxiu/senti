
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *

from senti.models.base.nn import NNBase

__all__ = ['NNShallow']


class NNShallow(NNBase):
    '''
    Trains the upper layers of an existing neural net.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._features = lambda X_batch: None

    def create_model(self, model, num_train):
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
        self.loss = -T.mean(lasagne.layers.get_output(l)[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    def gen_batches(self, X, y=None):
        model = self.kwargs['model']
        for X_batch, y_batch in model.gen_batches(X, y):
            yield (self._features(X_batch), y_batch)

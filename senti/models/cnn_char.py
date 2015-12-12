
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *

from senti.models.base.nn import NNBase
from senti.rand import get_rng
from senti.utils.lasagne_ import log_softmax

__all__ = ['CNNChar', 'CNNCharShallow']


class CNNChar(NNBase):
    def create_model(self, embeddings, input_size, output_size):
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        l = lasagne.layers.InputLayer((self.batch_size, input_size), self.inputs[0])
        l = lasagne.layers.EmbeddingLayer(l, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
        l = lasagne.layers.DimshuffleLayer(l, (0, 2, 1))
        conv_params = [(256, 7, 3), (256, 7, 3), (256, 3, None), (256, 3, None), (256, 3, None), (256, 3, 3)]
        for num_filters, filter_size, k in conv_params:
            l = lasagne.layers.Conv1DLayer(l, num_filters, filter_size, nonlinearity=rectify)
            if k is not None:
                l = lasagne.layers.MaxPool1DLayer(l, k, ignore_border=False)
        l = lasagne.layers.FlattenLayer(l)
        dense_params = [1024, 1024]
        for num_units in dense_params:
            l = lasagne.layers.DenseLayer(l, num_units, nonlinearity=rectify)
            l = lasagne.layers.DropoutLayer(l, 0.5)
        l = lasagne.layers.DenseLayer(l, 3, nonlinearity=log_softmax)
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = -T.mean(lasagne.layers.get_output(l)[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    def gen_batches(self, X, y=None):
        input_size = self.kwargs['input_size']
        X = np.vstack(np.hstack([x[input_size-1::-1], np.zeros(max(input_size - x.size, 0), dtype='int32')]) for x in X)
        n = X.shape[0]
        if y is None:
            indexes = np.hstack([np.arange(n), np.zeros(-n % self.batch_size, dtype='int32')])
        else:
            indexes = np.hstack([get_rng().permutation(n), get_rng().choice(n, -n % self.batch_size)])
        for i in range(0, indexes.size, self.batch_size):
            X_batch = X[indexes[i:i + self.batch_size]]
            y_batch = y[indexes[i:i + self.batch_size]] if y is not None else None
            yield (X_batch, y_batch)


class CNNCharShallow(NNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._features = None

    def create_model(self, model):
        self.inputs = [T.matrix('input')]
        self.target = T.ivector('target')
        l_feature = model.network.input_layer
        self._features = theano.function(model.inputs, lasagne.layers.get_output(l_feature, deterministic=True))
        l = lasagne.layers.InputLayer(lasagne.layers.get_output_shape(l_feature), self.inputs[0])
        l_feature.input_shape = l.shape
        l_feature.input_layer = l
        l = model.network
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = -T.mean(lasagne.layers.get_output(l)[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    # noinspection PyCallingNonCallable
    def gen_batches(self, X, y=None):
        model = self.kwargs['model']
        for X_batch, y_batch in model.gen_batches(X, y):
            yield (self._features(X_batch), y_batch)


import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *

from senti.models.nn_base import NNBase
from senti.rand import get_rng
from senti.utils.lasagne_ import log_softmax

__all__ = ['CNNChar']


class CNNChar(NNBase):
    def create_model(self, embeddings, input_size):
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        l = lasagne.layers.InputLayer((self.batch_size, input_size), self.inputs[0])
        l = lasagne.layers.EmbeddingLayer(l, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
        l = lasagne.layers.DimshuffleLayer(l, (0, 2, 1))
        conv_params = [(1024, 7, 3), (1024, 7, 3), (1024, 3, None), (1024, 3, None), (1024, 3, None), (1024, 3, 3)]
        for num_filters, filter_size, k in conv_params:
            l = lasagne.layers.Conv1DLayer(l, num_filters, filter_size, pad='full', nonlinearity=rectify)
            if k is not None:
                l = lasagne.layers.MaxPool1DLayer(l, k, ignore_border=False)
        dense_params = [2048, 2048]
        for num_units in dense_params:
            l = lasagne.layers.DenseLayer(l, num_units, nonlinearity=rectify)
            l = lasagne.layers.DropoutLayer(l, 0.5)
        l = lasagne.layers.DenseLayer(l, 3, nonlinearity=log_softmax)
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = -T.mean(lasagne.layers.get_output(l)[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    def gen_batches(self, X, y):
        input_size = self.kwargs['input_size']
        X = np.vstack(np.hstack([x[input_size-1::-1], np.zeros(max(input_size - x.size, 0), dtype='int32')]) for x in X)
        n = X.shape[0]
        if y is None:
            indexes = np.hstack([np.arange(n), np.zeros(-n % self.batch_size, dtype='int32')])
        else:
            indexes = np.hstack([get_rng().permutation(n), get_rng().choice(n, -n % self.batch_size)])
        for i in range(0, indexes.size, self.batch_size):
            X_batch = X[indexes[i:i + self.batch_size]]
            yield (X_batch,) if y is None else (X_batch, y[indexes[i:i + self.batch_size]])


import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *
from lasagne.objectives import *

from senti.models.base.nn import *

__all__ = ['RNNCharToWordEmbedding']


class RNNCharToWordEmbedding(NNRegressionBase):
    def create_model(self, emb_X, lstm_param, output_size):
        self.inputs = [T.imatrix('input'), T.matrix('mask')]
        self.target = T.matrix('target')
        l = lasagne.layers.InputLayer((self.batch_size, None), self.inputs[0])
        l_mask = lasagne.layers.InputLayer((self.batch_size, None), self.inputs[1])
        l = lasagne.layers.EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        l = lasagne.layers.LSTMLayer(l, lstm_param, nonlinearity=rectify, mask_input=l_mask)
        l = lasagne.layers.SliceLayer(l, -1, 1)
        l = lasagne.layers.DenseLayer(l, output_size, nonlinearity=identity)
        self.predictions = lasagne.layers.get_output(l, deterministic=True)
        self.loss = T.mean(aggregate(squared_error(lasagne.layers.get_output(l), self.target)))
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.rmsprop(self.loss, params, learning_rate=0.01)
        self.network = l

    def gen_batch(self, docs, y=None):
        shape = (len(docs), max(map(len, docs)))
        X = np.zeros(shape, dtype='int32')
        mask = np.zeros(shape, dtype='bool')
        for j, doc in enumerate(docs):
            X[j, :len(doc)] = doc
            mask[j, :len(doc)] = 1
        return X, mask, y

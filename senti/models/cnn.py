
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *

from senti.models.base.nn import NNBase
from senti.rand import get_rng
from senti.utils.lasagne_ import log_softmax

__all__ = ['CNN']


class CNN(NNBase):
    def create_model(self, embeddings, input_size, conv_param, dense_params, output_size, static_mode, norm_lim):
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        l = lasagne.layers.InputLayer((self.batch_size, input_size), self.inputs[0])
        l_embeds = []
        if static_mode in (0, 2):
            l_cur = lasagne.layers.EmbeddingLayer(l, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
            self.constraints[l_cur.W] = lambda u, v: u
            l_cur = lasagne.layers.DimshuffleLayer(l_cur, (0, 2, 1))
            l_embeds.append(l_cur)
        if static_mode in (1, 2):
            l_cur = lasagne.layers.EmbeddingLayer(l, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
            l_cur = lasagne.layers.DimshuffleLayer(l_cur, (0, 2, 1))
            l_embeds.append(l_cur)
        l_convs = []
        for filter_size in conv_param[1]:
            l_curs = [lasagne.layers.Conv1DLayer(
                l_embed, conv_param[0], filter_size, pad='full', nonlinearity=rectify
            ) for l_embed in l_embeds]
            l_cur = lasagne.layers.ElemwiseSumLayer(l_curs)
            l_cur = lasagne.layers.MaxPool1DLayer(l_cur, input_size + filter_size - 1, ignore_border=True)
            l_cur = lasagne.layers.FlattenLayer(l_cur)
            l_convs.append(l_cur)
        l = lasagne.layers.ConcatLayer(l_convs)
        l = lasagne.layers.DropoutLayer(l, 0.5)
        for dense_param in dense_params:
            l = lasagne.layers.DenseLayer(l, dense_param, nonlinearity=identity)
            self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, norm_lim)
            l = lasagne.layers.DropoutLayer(l, 0.5)
        l = lasagne.layers.DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, norm_lim)
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = -T.mean(lasagne.layers.get_output(l)[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    def gen_batches(self, X, y=None):
        X = np.vstack(np.hstack([x, np.zeros(self.kwargs['input_size'] - x.size, dtype='int32')]) for x in X)
        n = X.shape[0]
        if y is None:
            indexes = np.hstack([np.arange(n), np.zeros(-n % self.batch_size, dtype='int32')])
        else:
            indexes = np.hstack([get_rng().permutation(n), get_rng().choice(n, -n % self.batch_size)])
        for i in range(0, indexes.size, self.batch_size):
            X_batch = X[indexes[i:i + self.batch_size]]
            y_batch = y[indexes[i:i + self.batch_size]] if y is not None else None
            yield (X_batch, y_batch)

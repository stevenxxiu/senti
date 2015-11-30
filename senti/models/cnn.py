
import lasagne
import numpy as np
import theano.tensor as T

from senti.models.nn_base import NNBase
from senti.rand import get_rng
from senti.theano_utils import log_softmax

__all__ = ['CNN']


class CNN(NNBase):
    def create_model(
        self, embeddings, img_h, filter_hs, hidden_units, dropout_rates, conv_non_linear, activations, static_mode,
        lr_decay, norm_lim
    ):
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        l = lasagne.layers.InputLayer((self.batch_size, img_h), self.inputs[0])
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
        for filter_h in filter_hs:
            l_curs = [lasagne.layers.Conv1DLayer(
                l_embed, hidden_units[0], filter_h, pad='full', nonlinearity=conv_non_linear
            ) for l_embed in l_embeds]
            l_cur = lasagne.layers.ElemwiseSumLayer(l_curs)
            l_cur = lasagne.layers.MaxPool1DLayer(l_cur, img_h + filter_h - 1, ignore_border=True)
            l_cur = lasagne.layers.FlattenLayer(l_cur)
            l_convs.append(l_cur)
        l = lasagne.layers.ConcatLayer(l_convs)
        l = lasagne.layers.DropoutLayer(l, dropout_rates[0])
        for n, activation, dropout in zip(hidden_units[1:-1], activations, dropout_rates[1:]):
            l = lasagne.layers.DenseLayer(l, n, nonlinearity=activation)
            self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, norm_lim)
            l = lasagne.layers.DropoutLayer(l, dropout)
        l = lasagne.layers.DenseLayer(l, hidden_units[-1], nonlinearity=log_softmax)
        self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, norm_lim)
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = -T.mean(lasagne.layers.get_output(l)[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params, rho=lr_decay)
        self.network = l

    def gen_batches(self, X, y):
        n = X.shape[0]
        if y is None:
            indexes = np.hstack([np.arange(n), np.zeros(-n % self.batch_size, dtype='int32')])
        else:
            indexes = np.hstack([get_rng().permutation(n), get_rng().choice(n, -n % self.batch_size)])
        for i in range(0, indexes.size, self.batch_size):
            X_batch = X[indexes[i:i + self.batch_size]]
            yield (X_batch,) if y is None else (X_batch, y[indexes[i:i + self.batch_size]])

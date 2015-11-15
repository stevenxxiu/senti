
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import softmax

from senti.models.nn_base import NNBase
from senti.rand import get_rng

__all__ = ['CNN']


class CNN(NNBase):
    def create_model(
        self, embeddings, img_h, filter_hs, hidden_units, dropout_rates, conv_non_linear, activations, static_mode,
        lr_decay, norm_lim
    ):
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        net = lasagne.layers.InputLayer((self.batch_size, img_h), self.inputs[0])
        embedding_nets = []
        if static_mode in (0, 2):
            cur_net = lasagne.layers.EmbeddingLayer(net, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
            self.constraints[cur_net.W] = lambda u, v: u
            cur_net = lasagne.layers.DimshuffleLayer(cur_net, (0, 2, 1))
            embedding_nets.append(cur_net)
        if static_mode in (1, 2):
            cur_net = lasagne.layers.EmbeddingLayer(net, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
            cur_net = lasagne.layers.DimshuffleLayer(cur_net, (0, 2, 1))
            embedding_nets.append(cur_net)
        conv_nets = []
        for filter_h in filter_hs:
            cur_nets = [lasagne.layers.Conv1DLayer(
                cur_net, hidden_units[0], filter_h, pad='full', nonlinearity=conv_non_linear
            ) for cur_net in embedding_nets]
            cur_net = lasagne.layers.ElemwiseSumLayer(cur_nets)
            cur_net = lasagne.layers.MaxPool1DLayer(cur_net, img_h + filter_h - 1, ignore_border=True)
            cur_net = lasagne.layers.FlattenLayer(cur_net)
            conv_nets.append(cur_net)
        net = lasagne.layers.ConcatLayer(conv_nets)
        net = lasagne.layers.DropoutLayer(net, dropout_rates[0])
        for n, activation, dropout in zip(hidden_units[1:-1], activations, dropout_rates[1:]):
            net = lasagne.layers.DenseLayer(net, n, nonlinearity=activation)
            self.constraints[net.W] = lambda u, v: lasagne.updates.norm_constraint(v, norm_lim)
            net = lasagne.layers.DropoutLayer(net, dropout)
        net = lasagne.layers.DenseLayer(net, hidden_units[-1], nonlinearity=softmax)
        self.constraints[net.W] = lambda u, v: lasagne.updates.norm_constraint(v, norm_lim)
        self.probs = lasagne.layers.get_output(net, deterministic=True)
        self.loss = -T.mean(T.log(lasagne.layers.get_output(net))[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(net, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params, rho=lr_decay)
        self.network = net

    def gen_batches(self, X, y):
        n = X.shape[0]
        if y is None:
            indexes = np.hstack([np.arange(n), np.zeros(-n % self.batch_size, dtype='int32')])
        else:
            indexes = np.hstack([get_rng().permutation(n), get_rng().choice(n, -n % self.batch_size)])
        for i in range(0, len(indexes), self.batch_size):
            X_batch = X[indexes[i:i + self.batch_size]]
            yield (X_batch,) if y is None else (X_batch, y[indexes[i:i + self.batch_size]])

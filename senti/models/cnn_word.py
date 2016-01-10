
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *

from senti.utils.lasagne_ import *

__all__ = ['CNNWord', 'CNNWordPredInteraction']


class CNNWord(NNBase):
    def __init__(
        self, batch_size, emb_X, input_size, conv_param, dense_params, output_size, static_mode, max_norm, f1_classes
    ):
        super().__init__(batch_size)
        self.input_size = input_size
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        l = lasagne.layers.InputLayer((batch_size, input_size), self.inputs[0])
        l_embeds = []
        if static_mode in (0, 2):
            l_cur = lasagne.layers.EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
            self.constraints[l_cur.W] = lambda u, v: u
            l_cur = lasagne.layers.DimshuffleLayer(l_cur, (0, 2, 1))
            l_embeds.append(l_cur)
        if static_mode in (1, 2):
            l_cur = lasagne.layers.EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
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
        l = lasagne.layers.DropoutLayer(l)
        for dense_param in dense_params:
            l = lasagne.layers.DenseLayer(l, dense_param, nonlinearity=rectify)
            self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, max_norm)
            l = lasagne.layers.DropoutLayer(l)
        l = lasagne.layers.DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, max_norm)
        self.pred = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, lasagne.layers.get_output(l)))
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.metrics = {'train': [acc], 'dev': [acc, f1(f1_classes)]}
        self.network = l
        self.compile()

    def gen_batch(self, X, y=None):
        return np.vstack(np.pad(x[:self.input_size], (0, max(self.input_size - x.size, 0)), 'constant') for x in X), y


class CNNWordPredInteraction(NNBase):
    def __init__(
        self, batch_size, emb_X, input_size, conv_param, dense_params, output_size, max_norm, f1_classes
    ):
        super().__init__(batch_size)
        self.input_size = input_size
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        l = lasagne.layers.InputLayer((self.batch_size, input_size), self.inputs[0])
        l = lasagne.layers.EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        l = lasagne.layers.DimshuffleLayer(l, (0, 2, 1))
        l_convs = []
        for filter_size in conv_param[1]:
            l_cur = lasagne.layers.Conv1DLayer(l, conv_param[0], filter_size, pad='full', nonlinearity=rectify)
            l_cur = lasagne.layers.MaxPool1DLayer(l_cur, input_size + filter_size - 1, ignore_border=True)
            l_cur = lasagne.layers.FlattenLayer(l_cur)
            l_convs.append(l_cur)
        l = lasagne.layers.ConcatLayer(l_convs)
        l = lasagne.layers.DropoutLayer(l)
        for dense_param in dense_params:
            l = lasagne.layers.DenseLayer(l, dense_param, nonlinearity=rectify)
            self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, max_norm)
            l = lasagne.layers.DropoutLayer(l)
        l = lasagne.layers.DenseLayer(l, output_size, nonlinearity=identity)
        self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, max_norm)
        l = SelfInteractionLayer(l, nonlinearity=log_softmax)
        self.pred = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, lasagne.layers.get_output(l)))
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.metrics = {'train': [acc], 'dev': [acc, f1(f1_classes)]}
        self.network = l
        self.compile()

    def gen_batch(self, X, y=None):
        return np.vstack(np.pad(x[:self.input_size], (0, max(self.input_size - x.size, 0)), 'constant') for x in X), y

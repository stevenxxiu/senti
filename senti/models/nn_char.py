
import numpy as np
import theano.tensor as T
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.updates import *

from senti.utils.lasagne_ import *
from senti.utils.numpy_ import *

__all__ = ['CNNChar']


class CNNChar(NNBase):
    def __init__(self, batch_size, emb_X, input_size, output_size, static_mode, f1_classes):
        super().__init__(batch_size)
        self.input_size = input_size
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        l = InputLayer((batch_size, input_size), self.inputs[0])
        l = EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        if static_mode == 0:
            self.constraints[l.W] = lambda u, v: u
        l = DimshuffleLayer(l, (0, 2, 1))
        conv_params = [(256, 7, 3), (256, 7, 3), (256, 3, None), (256, 3, None), (256, 3, None), (256, 3, 3)]
        for num_filters, filter_size, k in conv_params:
            l = Conv1DLayer(l, num_filters, filter_size, nonlinearity=rectify)
            if k is not None:
                l = MaxPool1DLayer(l, k, ignore_border=False)
        l = FlattenLayer(l)
        dense_params = [(1024, 1), (1024, 20)]
        for num_units, max_norm in dense_params:
            l = DenseLayer(l, num_units, nonlinearity=rectify)
            if max_norm:
                self.constraints[l.W] = lambda u, v: norm_constraint(v, max_norm)
            l = DropoutLayer(l)
        l = DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.probs = T.exp(get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, get_output(l)))
        params = get_all_params(l, trainable=True)
        self.updates = adadelta(self.loss, params)
        self.metrics = {'train': [acc], 'dev': [acc, f1(f1_classes)]}
        self.network = l
        self.compile()

    def gen_batch(self, docs, y=None):
        return np.vstack(clippad(doc[::-1], self.input_size) for doc in docs), y

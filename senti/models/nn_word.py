
import numpy as np
import theano.tensor as T
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.updates import *

from senti.utils.lasagne_ import *

__all__ = ['CNNWord', 'CNNWordPredInteraction', 'RNNWord']


class CNNWord(NNBase):
    def __init__(
        self, batch_size, emb_X, input_size, conv_param, dense_params, output_size, static_mode, max_norm, f1_classes
    ):
        super().__init__(batch_size)
        self.input_size = input_size
        self.inputs = [T.imatrix('input')]
        self.target = T.ivector('target')
        l = InputLayer((batch_size, input_size), self.inputs[0])
        l_embeds = []
        if static_mode in (0, 2):
            l_cur = EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
            self.constraints[l_cur.W] = lambda u, v: u
            l_cur = DimshuffleLayer(l_cur, (0, 2, 1))
            l_embeds.append(l_cur)
        if static_mode in (1, 2):
            l_cur = EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
            l_cur = DimshuffleLayer(l_cur, (0, 2, 1))
            l_embeds.append(l_cur)
        l_convs = []
        for filter_size in conv_param[1]:
            l_curs = [Conv1DLayer(
                l_embed, conv_param[0], filter_size, pad='full', nonlinearity=rectify
            ) for l_embed in l_embeds]
            l_cur = ElemwiseSumLayer(l_curs)
            l_cur = MaxPool1DLayer(l_cur, input_size + filter_size - 1, ignore_border=True)
            l_cur = FlattenLayer(l_cur)
            l_convs.append(l_cur)
        l = ConcatLayer(l_convs)
        l = DropoutLayer(l)
        for dense_param in dense_params:
            l = DenseLayer(l, dense_param, nonlinearity=rectify)
            self.constraints[l.W] = lambda u, v: norm_constraint(v, max_norm)
            l = DropoutLayer(l)
        l = DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.constraints[l.W] = lambda u, v: norm_constraint(v, max_norm)
        self.pred = T.exp(get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, get_output(l)))
        params = get_all_params(l, trainable=True)
        self.updates = adadelta(self.loss, params)
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
        l = InputLayer((self.batch_size, input_size), self.inputs[0])
        l = EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        l = DimshuffleLayer(l, (0, 2, 1))
        l_convs = []
        for filter_size in conv_param[1]:
            l_cur = Conv1DLayer(l, conv_param[0], filter_size, pad='full', nonlinearity=rectify)
            l_cur = MaxPool1DLayer(l_cur, input_size + filter_size - 1, ignore_border=True)
            l_cur = FlattenLayer(l_cur)
            l_convs.append(l_cur)
        l = ConcatLayer(l_convs)
        l = DropoutLayer(l)
        for dense_param in dense_params:
            l = DenseLayer(l, dense_param, nonlinearity=rectify)
            self.constraints[l.W] = lambda u, v: norm_constraint(v, max_norm)
            l = DropoutLayer(l)
        l = DenseLayer(l, output_size, nonlinearity=identity)
        self.constraints[l.W] = lambda u, v: norm_constraint(v, max_norm)
        l = SelfInteractionLayer(l, nonlinearity=log_softmax)
        self.pred = T.exp(get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, get_output(l)))
        params = get_all_params(l, trainable=True)
        self.updates = adadelta(self.loss, params)
        self.metrics = {'train': [acc], 'dev': [acc, f1(f1_classes)]}
        self.network = l
        self.compile()

    def gen_batch(self, X, y=None):
        return np.vstack(np.pad(x[:self.input_size], (0, max(self.input_size - x.size, 0)), 'constant') for x in X), y


class RNNWord(NNBase):
    def __init__(self, batch_size, emb_X, lstm_param, output_size, f1_classes):
        super().__init__(batch_size)
        self.inputs = [T.imatrix('input'), T.matrix('mask')]
        self.target = T.ivector('target')
        l = InputLayer((batch_size, None), self.inputs[0])
        l_mask = InputLayer((batch_size, None), self.inputs[1])
        l = EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        l = LSTMLayer(
            l, lstm_param, mask_input=l_mask, grad_clipping=100, nonlinearity=rectify,
            only_return_final=True
        )
        l = DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.pred = T.exp(get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, get_output(l)))
        params = get_all_params(l, trainable=True)
        self.updates = rmsprop(self.loss, params, learning_rate=0.01)
        self.metrics = {'train': [acc], 'dev': [acc, f1(f1_classes)]}
        self.network = l
        self.compile()

    def gen_batch(self, docs, y=None):
        shape_ = (len(docs), max(map(len, docs)))
        X = np.zeros(shape_, dtype='int32')
        mask = np.zeros(shape_, dtype='bool')
        for i, doc in enumerate(docs):
            X[i, :len(doc)] = doc
            mask[i, :len(doc)] = 1
        return X, mask, y

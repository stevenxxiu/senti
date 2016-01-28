
import numpy as np
import theano.tensor as T
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *

from senti.utils.lasagne_ import *

__all__ = ['RNNCharToWordEmbedding', 'RNNCharCNNWord']


class RNNCharToWordEmbedding(NNBase):
    def __init__(self, batch_size, emb_X, lstm_params, output_size):
        super().__init__(batch_size)
        self.inputs = [T.imatrix('input'), T.matrix('mask')]
        self.target = T.matrix('target')
        l = InputLayer((batch_size, None), self.inputs[0])
        l_mask = InputLayer((batch_size, None), self.inputs[1])
        l = EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        for lstm_param in lstm_params:
            l = LSTMLayer(
                l, lstm_param, grad_clipping=100, nonlinearity=tanh, mask_input=l_mask, only_return_final=True
            )
        l = DenseLayer(l, output_size, nonlinearity=identity)
        self.pred = get_output(l, deterministic=True)
        self.loss = T.mean(aggregate(squared_error(get_output(l), self.target)))
        params = get_all_params(l, trainable=True)
        self.update_params = [T.scalar('learning_rate')]
        self.updates = rmsprop(self.loss, params, learning_rate=self.update_params[0])
        self.metrics = {'train': [rmse], 'val': [rmse]}
        self.network = l
        self.compile()

    def gen_batch(self, docs, y=None):
        shape = (len(docs), max(map(len, docs)))
        X = np.zeros(shape, dtype='int32')
        mask = np.zeros(shape, dtype='bool')
        for j, doc in enumerate(docs):
            X[j, :len(doc)] = doc
            mask[j, :len(doc)] = 1
        return X, mask, y


class RNNCharCNNWord(NNBase):
    def __init__(self, batch_size, emb_X, num_words, lstm_params, conv_param, output_size, f1_classes):
        super().__init__(batch_size)
        self.num_words = num_words
        self.inputs = [T.itensor3('input'), T.tensor3('mask')]
        self.target = T.ivector('target')
        l = InputLayer((batch_size, num_words, None), self.inputs[0])
        l_mask = InputLayer((batch_size, num_words, None), self.inputs[1])
        l = ReshapeLayer(l, (-1, [2]))
        l_mask = ReshapeLayer(l_mask, (-1, [2]))
        l = EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        for lstm_param in lstm_params:
            l = LSTMLayer(
                l, lstm_param, grad_clipping=100, nonlinearity=tanh, mask_input=l_mask, only_return_final=True
            )
        l = ReshapeLayer(l, (batch_size, num_words, -1))
        l_convs = []
        for filter_size in conv_param[1]:
            l_cur = Conv1DLayer(l, conv_param[0], filter_size, pad='full', nonlinearity=rectify)
            l_cur = MaxPool1DLayer(l_cur, num_words + filter_size - 1, ignore_border=True)
            l_cur = FlattenLayer(l_cur)
            l_convs.append(l_cur)
        l = ConcatLayer(l_convs)
        l = DropoutLayer(l)
        l = DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.constraints[l.W] = lambda u, v: norm_constraint(v, 3)
        self.pred = T.exp(get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, get_output(l)))
        params = get_all_params(l, trainable=True)
        self.updates = adadelta(self.loss, params)
        self.metrics = {'train': [acc], 'val': [acc, f1(f1_classes)]}
        self.network = l
        self.compile()

    def gen_batch(self, docs, y=None):
        docs = list(map(list, docs))  # make deterministic
        shape = (len(docs), self.num_words, max(len(word) for doc in docs for word in doc))
        X = np.zeros(shape, dtype='int32')
        mask = np.zeros(shape, dtype='bool')
        for i, doc in enumerate(docs):
            for j, word in zip(range(self.num_words), doc):
                X[i, j, :len(word)] = word
                mask[i, j, :len(word)] = 1
        return X, mask, y

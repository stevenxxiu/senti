
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import *
from lasagne.objectives import *

from senti.models.base.nn import *
from senti.utils.lasagne_ import *

__all__ = ['RNNCharToWordEmbedding', 'RNNCharCNNWord']


class RNNCharToWordEmbedding(NNRegressorBase):
    def create_model(self, emb_X, lstm_params, output_size):
        self.inputs = [T.imatrix('input'), T.matrix('mask')]
        self.target = T.matrix('target')
        l = lasagne.layers.InputLayer((self.batch_size, None), self.inputs[0])
        l_mask = lasagne.layers.InputLayer((self.batch_size, None), self.inputs[1])
        l = lasagne.layers.EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        for lstm_param in lstm_params:
            l = lasagne.layers.LSTMLayer(l, lstm_param, grad_clipping=100, nonlinearity=tanh, mask_input=l_mask)
        l = lasagne.layers.SliceLayer(l, -1, 1)
        l = lasagne.layers.DenseLayer(l, output_size, nonlinearity=identity)
        self.predictions = lasagne.layers.get_output(l, deterministic=True)
        self.loss = T.mean(aggregate(squared_error(lasagne.layers.get_output(l), self.target)))
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.update_params = [T.scalar('learning_rate')]
        self.updates = lasagne.updates.rmsprop(self.loss, params, learning_rate=self.update_params[0])
        self.network = l

    def gen_batch(self, docs, y=None):
        shape = (len(docs), max(map(len, docs)))
        X = np.zeros(shape, dtype='int32')
        mask = np.zeros(shape, dtype='bool')
        for j, doc in enumerate(docs):
            X[j, :len(doc)] = doc
            mask[j, :len(doc)] = 1
        return X, mask, y


class RNNCharCNNWord(NNClassifierBase):
    def create_model(self, emb_X, num_words, lstm_params, conv_param, output_size):
        self.inputs = [T.itensor3('input'), T.tensor3('mask')]
        self.target = T.ivector('target')
        l = lasagne.layers.InputLayer((self.batch_size, num_words, None), self.inputs[0])
        l_mask = lasagne.layers.InputLayer((self.batch_size, num_words, None), self.inputs[1])
        l = lasagne.layers.ReshapeLayer(l, (-1, [2]))
        l_mask = lasagne.layers.ReshapeLayer(l_mask, (-1, [2]))
        l = lasagne.layers.EmbeddingLayer(l, emb_X.shape[0], emb_X.shape[1], W=emb_X)
        for lstm_param in lstm_params:
            l = lasagne.layers.LSTMLayer(l, lstm_param, grad_clipping=100, nonlinearity=tanh, mask_input=l_mask)
        l = lasagne.layers.SliceLayer(l, -1, 1)
        l = lasagne.layers.ReshapeLayer(l, (self.batch_size, num_words, -1))
        l_convs = []
        for filter_size in conv_param[1]:
            l_cur = lasagne.layers.Conv1DLayer(l, conv_param[0], filter_size, pad='full', nonlinearity=rectify)
            l_cur = lasagne.layers.MaxPool1DLayer(l_cur, num_words + filter_size - 1, ignore_border=True)
            l_cur = lasagne.layers.FlattenLayer(l_cur)
            l_convs.append(l_cur)
        l = lasagne.layers.ConcatLayer(l_convs)
        l = lasagne.layers.DropoutLayer(l)
        l = lasagne.layers.DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.constraints[l.W] = lambda u, v: lasagne.updates.norm_constraint(v, 3)
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(lasagne.layers.get_output(l), self.target, self.batch_size))
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    def gen_batch(self, docs, y=None):
        num_words = self.kwargs['num_words']
        docs = list(map(list, docs))  # make deterministic
        shape = (len(docs), num_words, max(len(word) for doc in docs for word in doc))
        X = np.zeros(shape, dtype='int32')
        mask = np.zeros(shape, dtype='bool')
        for i, doc in enumerate(docs):
            for j, word in zip(range(num_words), doc):
                X[i, j, :len(word)] = word
                mask[i, j, :len(word)] = 1
        return X, mask, y


import theano.tensor as T
from lasagne.init import *
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.updates import *

from senti.utils.lasagne_ import *

__all__ = ['RNNWord']


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

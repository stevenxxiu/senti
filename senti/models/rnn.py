
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.nonlinearities import rectify

from senti.models.nn_base import NNBase
from senti.rand import get_rng
from senti.theano_utils import log_softmax

__all__ = ['RNN']


class RNN(NNBase):
    def create_model(self, embeddings, hidden_units):
        self.inputs = [T.imatrix('input'), T.matrix('mask')]
        self.target = T.ivector('target')
        l = lasagne.layers.InputLayer((self.batch_size, None), self.inputs[0])
        l_mask = lasagne.layers.InputLayer((self.batch_size, None), self.inputs[1])
        l = lasagne.layers.EmbeddingLayer(l, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
        l = lasagne.layers.LSTMLayer(l, hidden_units[0], nonlinearity=rectify, mask_input=l_mask)
        l = lasagne.layers.SliceLayer(l, -1, 1)
        l = lasagne.layers.DenseLayer(l, hidden_units[-1], nonlinearity=log_softmax)
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = -T.mean(lasagne.layers.get_output(l)[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    def gen_batches(self, docs, y):
        docs = list(docs)
        n = len(docs)
        if y is None:
            indexes = np.hstack([np.arange(n), np.zeros(-n % self.batch_size, dtype='int32')])
        else:
            indexes = np.hstack([get_rng().permutation(n), get_rng().choice(n, -n % self.batch_size)])
        for i in range(0, indexes.size, self.batch_size):
            cur_docs = [docs[indexes[j]] for j in range(i, i + self.batch_size)]
            shape = (self.batch_size, max(map(len, cur_docs)))
            X_batch, mask_batch = np.zeros(shape, dtype='int32'), np.zeros(shape, dtype='bool')
            for j, doc in enumerate(cur_docs):
                X_batch[j, :len(doc)] = doc
                mask_batch[j, :len(doc)] = 1
            yield (X_batch, mask_batch) if y is None else (X_batch, mask_batch, y[indexes[i:i + self.batch_size]])

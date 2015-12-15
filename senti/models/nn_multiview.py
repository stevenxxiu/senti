
import lasagne
import numpy as np
import theano.tensor as T

from senti.models.base.nn import NNBase
from senti.utils.lasagne_ import log_softmax

__all__ = ['NNMultiView']


class NNMultiView(NNBase):
    def create_model(self, models, output_size):
        self.inputs = sum((model.inputs for model in models), [])
        self.target = T.ivector('target')
        l_features = []
        for model in models:
            l_features.append(model.network.input_layer)
        l = lasagne.layers.ConcatLayer(l_features)
        l = lasagne.layers.DropoutLayer(l)
        l = lasagne.layers.DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.probs = T.exp(lasagne.layers.get_output(l, deterministic=True))
        self.loss = -T.mean(lasagne.layers.get_output(l)[np.arange(self.batch_size), self.target])
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    def gen_batch(self, Xs, y=None):
        models = self.kwargs['models']
        return [*(model.gen_batch(X, y)[0] for X, model in zip(zip(*Xs), models)), y]


import lasagne
import theano.tensor as T

from senti.models.base.nn import *
from senti.utils.lasagne_ import *

__all__ = ['NNMultiView']


class NNMultiView(NNClassifierBase):
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
        self.loss = T.mean(categorical_crossentropy_exp(lasagne.layers.get_output(l), self.target, self.batch_size))
        params = lasagne.layers.get_all_params(l, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params)
        self.network = l

    def gen_batch(self, Xs, y=None):
        models = self.kwargs['models']
        return [*(model.gen_batch(X, y)[0] for X, model in zip(zip(*Xs), models)), y]

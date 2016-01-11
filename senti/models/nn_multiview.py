
import theano.tensor as T
from lasagne.layers import *
from lasagne.updates import *

from senti.utils.lasagne_ import *

__all__ = ['NNMultiView']


class NNMultiView(NNBase):
    def __init__(self, models_, output_size):
        super().__init__(models_[0].batch_size)
        self.models = models_
        self.inputs = sum((model.inputs for model in models_), [])
        self.target = T.ivector('target')
        l_features = []
        for model in models_:
            l_features.append(model.network.input_layer)
        l = ConcatLayer(l_features)
        l = DropoutLayer(l)
        l = DenseLayer(l, output_size, nonlinearity=log_softmax)
        self.pred = T.exp(get_output(l, deterministic=True))
        self.loss = T.mean(categorical_crossentropy_exp(self.target, get_output(l)))
        params = get_all_params(l, trainable=True)
        self.updates = adadelta(self.loss, params)
        self.metrics = models_[0].metrics
        self.network = l
        self.compile()

    def gen_batch(self, Xs, y=None):
        return [*(model.gen_batch(X, y)[0] for X, model in zip(zip(*Xs), self.models)), y]

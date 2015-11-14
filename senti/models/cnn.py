
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.nonlinearities import softmax
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from senti.rand import get_rng

__all__ = ['ConvNet']


def pad_docs(X, y, batch_size, rand=False):
    n_extra = (-X.shape[0]) % batch_size
    if n_extra > 0:
        if rand:
            indexes = get_rng().choice(X.shape[0], n_extra, replace=False)
            X_extra, y_extra = X[indexes], y[indexes]
        else:
            X_extra, y_extra = np.zeros((n_extra, X.shape[1]), dtype=X.dtype), np.zeros(n_extra, dtype=y.dtype)
        X, y = np.vstack([X, X_extra]), np.concatenate([y, y_extra])
    return X, y, X.shape[0]//batch_size


class ConvNet(BaseEstimator):
    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        self.args = args
        self.kwargs = kwargs
        self.classes_ = None
        self.network = None

    def _create_model(
        self, embeddings, img_h, filter_hs, hidden_units, dropout_rates, conv_non_linear, activations, static_mode,
        lr_decay, norm_lim
    ):
        constraints = {}
        self.X = T.imatrix('X')
        self.y = T.ivector('y')
        net = lasagne.layers.InputLayer((self.batch_size, img_h), self.X)
        embedding_nets = []
        if static_mode in (0, 2):
            cur_net = lasagne.layers.EmbeddingLayer(net, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
            constraints[cur_net.W] = lambda u, v: u
            cur_net = lasagne.layers.DimshuffleLayer(cur_net, (0, 2, 1))
            embedding_nets.append(cur_net)
        if static_mode in (1, 2):
            cur_net = lasagne.layers.EmbeddingLayer(net, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
            cur_net = lasagne.layers.DimshuffleLayer(cur_net, (0, 2, 1))
            embedding_nets.append(cur_net)
        conv_nets = []
        for filter_h in filter_hs:
            cur_nets = [lasagne.layers.Conv1DLayer(
                cur_net, hidden_units[0], filter_h, pad='full', nonlinearity=conv_non_linear
            ) for cur_net in embedding_nets]
            cur_net = lasagne.layers.ElemwiseSumLayer(cur_nets)
            cur_net = lasagne.layers.MaxPool1DLayer(cur_net, img_h + filter_h - 1, ignore_border=True)
            cur_net = lasagne.layers.FlattenLayer(cur_net)
            conv_nets.append(cur_net)
        net = lasagne.layers.ConcatLayer(conv_nets)
        net = lasagne.layers.DropoutLayer(net, dropout_rates[0])
        for n, activation, dropout in zip(hidden_units[1:-1], activations, dropout_rates[1:]):
            net = lasagne.layers.DenseLayer(net, n, nonlinearity=activation)
            constraints[net.W] = lambda u, v: lasagne.updates.norm_constraint(v, norm_lim)
            net = lasagne.layers.DropoutLayer(net, dropout)
        net = lasagne.layers.DenseLayer(net, hidden_units[-1], nonlinearity=softmax)
        constraints[net.W] = lambda u, v: lasagne.updates.norm_constraint(v, norm_lim)
        self.network = net
        self.prediction_probs = lasagne.layers.get_output(self.network, deterministic=True)
        self.loss = -T.mean(T.log(lasagne.layers.get_output(net))[T.arange(self.y.shape[0]), self.y])
        params = lasagne.layers.get_all_params(net, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params, rho=lr_decay)
        for param, constraint in constraints.items():
            self.updates[param] = constraint(param, self.updates[param])

    def fit(self, X, y, shuffle_batch, n_epochs, dev_X, dev_y, average_classes):
        if not self.network:
            self._create_model(*self.args, **self.kwargs)
        self.classes_ = unique_labels(dev_y)

        # process & load into shared variables to allow theano to copy all data to GPU memory for speed
        n_train_docs, n_dev_docs = X.shape[0], dev_X.shape[0]
        indexes = get_rng().permutation(n_train_docs)
        train_X, train_y = X[indexes], y[indexes]
        train_X, train_y, n_train_batches = pad_docs(train_X, train_y, self.batch_size, rand=True)
        dev_X, _, n_dev_batches = pad_docs(dev_X, np.empty(0), self.batch_size)
        train_X, train_y = theano.shared(np.int32(train_X), borrow=True), theano.shared(np.int32(train_y), borrow=True)
        dev_X = theano.shared(np.int32(dev_X), borrow=True)

        # theano functions
        index = T.lscalar()
        predictions = T.argmax(self.prediction_probs, axis=1)
        acc = T.mean(T.eq(predictions, self.y))
        train_batch = theano.function([index], [self.loss, acc], updates=self.updates, givens={
            self.X: train_X[index*self.batch_size:(index + 1)*self.batch_size],
            self.y: train_y[index*self.batch_size:(index + 1)*self.batch_size]
        })
        test_batch = theano.function([index], predictions, givens={
            self.X: dev_X[index*self.batch_size:(index + 1)*self.batch_size]
        })

        # start training over mini-batches
        print('training cnn...')
        best_perf = 0
        params = lasagne.layers.get_all_params(self.network)
        best_params = None
        for epoch in range(n_epochs):
            batch_indices = np.arange(n_train_batches)
            if shuffle_batch:
                get_rng().shuffle(batch_indices)
            train_res = []
            for i in batch_indices:
                train_res.append(train_batch(i))
            train_loss, train_acc = np.mean(train_res, axis=0)
            dev_res = np.hstack((test_batch(i) for i in range(n_dev_batches)))[:n_dev_docs]
            dev_acc = accuracy_score(dev_res, dev_y)
            dev_f1 = np.mean(precision_recall_fscore_support(dev_res, dev_y)[2][average_classes])
            if dev_f1 >= best_perf:
                best_perf = dev_f1
                best_params = {param: param.get_value() for param in params}
            print('epoch {}, train loss {:.4f}, train acc {:.4f}, val acc {:.4f}, val f1 {:.4f}'.format(
                epoch + 1, train_loss, train_acc, dev_acc, dev_f1
            ))
        for param, value in best_params.items():
            param.set_value(value)

    def predict_proba(self, X):
        n_docs = X.shape[0]
        X, _, n_batches = pad_docs(X, np.empty(0), self.batch_size)
        X = theano.shared(np.int32(X), borrow=True)
        index = T.lscalar()
        predict_batch = theano.function([index], self.prediction_probs, givens={
            self.X: X[index*self.batch_size:(index + 1)*self.batch_size]
        })
        return np.vstack(predict_batch(i) for i in range(n_batches))[:n_docs]

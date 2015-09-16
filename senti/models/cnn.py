
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.nonlinearities import softmax
from lasagne.regularization import apply_penalty, l2, l1
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from senti.rand import get_rng

__all__ = ['ConvNet']


def pad_docs(X, y, batch_size, rand=False):
    n_extra = (-X.shape[0]) % batch_size
    if n_extra == 0:
        return
    if rand:
        indexes = get_rng().choice(X.shape[0], n_extra, replace=False)
        X_extra, y_extra = X[indexes], y[indexes]
    else:
        X_extra, y_extra = np.zeros((n_extra, X.shape[1]), dtype=X.dtype), np.zeros(n_extra, dtype=y.dtype)
    return np.vstack([X, X_extra]), np.concatenate([y, y_extra]), (X.shape[0] + n_extra)//batch_size


class ConvNet(BaseEstimator):
    def __init__(self, batch_size, shuffle_batch, n_epochs, dev_X, dev_y, average_classes, *args, **kwargs):
        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch
        self.n_epochs = n_epochs
        self.dev_X = dev_X
        self.dev_y = dev_y
        self.average_classes = average_classes
        self.classes_ = unique_labels(dev_y)
        self.args = args
        self.kwargs = kwargs

    def _create_model(
        self, embeddings, img_h, filter_hs, hidden_units, dropout_rates, conv_non_linear, activations, non_static,
        lr_decay
    ):
        constraints = {}
        penalty = 0
        self.X = T.imatrix('X')
        self.y = T.ivector('y')
        network = lasagne.layers.InputLayer((self.batch_size, img_h), self.X)
        network = lasagne.layers.EmbeddingLayer(network, embeddings.X.shape[0], embeddings.X.shape[1], W=embeddings.X)
        if not non_static:
            constraints[network.W] = lambda v: network.W
        network = lasagne.layers.DimshuffleLayer(network, (0, 2, 1))
        convs = []
        for filter_h in filter_hs:
            conv = network
            conv = lasagne.layers.Conv1DLayer(conv, hidden_units[0], filter_h, pad='full', nonlinearity=conv_non_linear)
            penalty += apply_penalty((conv.W, conv.b), l2)*5e-4
            conv = lasagne.layers.MaxPool1DLayer(conv, img_h + filter_h - 1, ignore_border=True)
            conv = lasagne.layers.FlattenLayer(conv)
            convs.append(conv)
        network = lasagne.layers.ConcatLayer(convs)
        network = lasagne.layers.DropoutLayer(network, dropout_rates[0])
        for n, activation, dropout in zip(hidden_units[1:-1], activations, dropout_rates[1:]):
            network = lasagne.layers.DenseLayer(network, n, nonlinearity=activation)
            network = lasagne.layers.DropoutLayer(network, dropout)
        network = lasagne.layers.DenseLayer(network, hidden_units[-1], nonlinearity=softmax)
        penalty += apply_penalty((network.W, network.b), l1)*5e-4
        self.network = network
        self.prediction_probs = lasagne.layers.get_output(self.network, deterministic=True)
        # regularized cross-entropy
        params = lasagne.layers.get_all_params(network, trainable=True)
        self.loss = -T.mean(T.log(lasagne.layers.get_output(network))[T.arange(self.y.shape[0]), self.y])
        self.loss += penalty
        self.updates = lasagne.updates.adadelta(self.loss, params, rho=lr_decay)
        for param, constraint in constraints.items():
            self.updates[param] = constraint(self.updates[param])

    def fit(self, X, y):
        self._create_model(*self.args, **self.kwargs)

        # process & load into shared variables to allow theano to copy all data to GPU memory for speed
        n_train_docs, n_dev_docs = X.shape[0], self.dev_X.shape[0]
        indexes = get_rng().permutation(n_train_docs)
        train_X, train_y = X[indexes], y[indexes]
        train_X, train_y, n_train_batches = pad_docs(train_X, train_y, self.batch_size, rand=True)
        dev_X, _, n_dev_batches = pad_docs(self.dev_X, np.empty(0), self.batch_size)
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
        for epoch in range(self.n_epochs):
            batch_indices = np.arange(n_train_batches)
            if self.shuffle_batch:
                get_rng().shuffle(batch_indices)
            train_res = []
            for i in batch_indices:
                train_res.append(train_batch(i))
            train_loss, train_acc = np.mean(train_res, axis=0)
            dev_res = np.hstack((test_batch(i) for i in range(n_dev_batches)))[:n_dev_docs]
            print('epoch {}, train loss {:.4f}, train acc {:.4f}, val acc {:.4f}, val f1 {:.4f}'.format(
                epoch + 1, train_loss, train_acc, accuracy_score(dev_res, self.dev_y),
                np.mean(precision_recall_fscore_support(dev_res, self.dev_y)[2][self.average_classes])
            ))

    def predict_proba(self, X):
        n_docs = X.shape[0]
        X, _, n_batches = pad_docs(X, np.empty(0), self.batch_size)
        X = theano.shared(np.int32(X), borrow=True)
        index = T.lscalar()
        predict_batch = theano.function([index], self.prediction_probs, givens={
            self.X: X[index*self.batch_size:(index + 1)*self.batch_size]
        })
        return np.vstack(predict_batch(i) for i in range(n_batches))[:n_docs]

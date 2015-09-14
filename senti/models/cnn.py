
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.nonlinearities import softmax
from sklearn.base import BaseEstimator

__all__ = ['ConvNet']


class ConvNet(BaseEstimator):
    def __init__(
        self, word_vecs, img_w, img_h, filter_hs, hidden_units, dropout_rates, conv_non_linear, activations, non_static,
        shuffle_batch, n_epochs, batch_size, lr_decay, norm_lim
    ):
        '''
        args:
            img_w: word vector length (300 for word2vec)
            img_h: sentence length (padded where necessary)
            filter_hs: filter window sizes
            hidden_units: [x, y] x is the number of feature maps (per filter window), and y is the penultimate layer
            sqr_norm_lim: s^2 in the paper
            lr_decay: adadelta decay parameter
        '''
        self.img_w = img_w
        self.img_h = img_h
        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch
        self.n_epochs = n_epochs
        self.classes_ = np.arange(hidden_units[-1])
        constraints = {}
        self.X = T.imatrix('X')
        self.y = T.ivector('y')
        network = lasagne.layers.InputLayer((batch_size, img_h), self.X)
        network = lasagne.layers.EmbeddingLayer(network, word_vecs.shape[0], word_vecs.shape[1], W=word_vecs)
        if not non_static:
            constraints[network.W] = lambda v: network.W
        network = lasagne.layers.DimshuffleLayer(network, (0, 2, 1))
        convs = []
        for filter_h in filter_hs:
            conv = network
            conv = lasagne.layers.Conv1DLayer(conv, hidden_units[0], filter_h, pad='full', nonlinearity=conv_non_linear)
            conv = lasagne.layers.MaxPool1DLayer(conv, img_h + filter_h - 1, ignore_border=True)
            conv = lasagne.layers.FlattenLayer(conv)
            convs.append(conv)
        network = lasagne.layers.ConcatLayer(convs)
        network = lasagne.layers.DropoutLayer(network, dropout_rates[0])
        for n, activation, dropout in zip(hidden_units[1:-1], activations, dropout_rates[1:]):
            network = lasagne.layers.DenseLayer(network, n, nonlinearity=activation)
            constraints[network.W] = lambda v: lasagne.updates.norm_constraint(v, norm_lim)
            network = lasagne.layers.DropoutLayer(network, dropout)
        network = lasagne.layers.DenseLayer(network, hidden_units[-1], nonlinearity=softmax)
        constraints[network.W] = lambda v: lasagne.updates.norm_constraint(v, norm_lim)
        self.network = network
        self.prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.loss = -T.mean(T.log(lasagne.layers.get_output(network))[T.arange(self.y.shape[0]), self.y])
        params = lasagne.layers.get_all_params(network, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, params, rho=lr_decay)
        for param, constraint in constraints.items():
            self.updates[param] = constraint(self.updates[param])

    def fit(self, X, y):
        np.random.seed(3435)
        dataset = np.hstack([X, y.reshape((-1, 1))])
        num_docs = dataset.shape[0]

        # Shuffle dataset and assign to mini batches. If dataset size is not a multiple of mini batches, replicate
        # extra data (at random).
        if num_docs % self.batch_size > 0:
            extra_data_num = self.batch_size - num_docs % self.batch_size
            dataset = np.vstack([dataset, dataset[np.random.choice(num_docs, extra_data_num, replace=False)]])
        dataset = np.random.permutation(dataset)
        n_batches = num_docs//self.batch_size
        n_train_batches = round(n_batches*0.9)

        # divide train set into train/val sets
        train_set = dataset[:n_train_batches*self.batch_size, :]
        val_set = dataset[n_train_batches*self.batch_size:, :]
        train_set_x, train_set_y = shared_dataset((train_set[:, :self.img_h], train_set[:, -1]))
        val_set_x, val_set_y = shared_dataset((val_set[:, :self.img_h], val_set[:, -1]))
        n_val_batches = n_batches - n_train_batches

        # theano functions
        index = T.lscalar()
        acc = T.mean(T.eq(T.argmax(self.prediction, axis=1), self.y), dtype=theano.config.floatX)
        train_batch = theano.function([index], [self.loss, acc], updates=self.updates, givens={
            self.X: train_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            self.y: train_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })
        test_batch = theano.function([index], acc, givens={
            self.X: val_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            self.y: val_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })

        # start training over mini-batches
        print('training cnn...')
        for epoch in range(self.n_epochs):
            batch_indices = np.arange(n_train_batches)
            if self.shuffle_batch:
                np.random.shuffle(batch_indices)
            train_res = []
            for i in batch_indices:
                train_res.append(train_batch(i))
            train_res = np.mean(train_res, axis=0)
            test_res = np.mean(list(test_batch(i) for i in range(n_val_batches)))
            print('epoch {}, train loss {}, train perf {} %, val perf {} %'.format(
                epoch + 1, train_res[0]*100, train_res[1]*100, test_res*100
            ))

    def predict_proba(self, X):
        n_docs = X.shape[0]
        n_batches = (n_docs + self.batch_size - 1)//self.batch_size
        X = np.vstack([X, np.zeros((n_batches*self.batch_size - n_docs, X.shape[1]), dtype=X.dtype)])
        X = theano.shared(X, borrow=True)
        index = T.lscalar()
        predict_batch = theano.function([index], self.prediction, givens={
            self.X: X[index*self.batch_size:(index + 1)*self.batch_size]
        })
        return np.vstack(predict_batch(i) for i in range(n_batches))[:n_docs]


def shared_dataset(data_xy):
    '''
    Function that loads the dataset into shared variables. The reason we store our dataset in shared variables is to
    allow Theano to copy it into the GPU memory (when code is run on GPU). Since copying data into the GPU is slow,
    copying a minibatch everytime is needed (the default behaviour if the data is not in a shared variable) would lead
    to a large decrease in performance.
    '''
    data_x, data_y = data_xy
    shared_x = theano.shared(data_x, borrow=True)
    shared_y = theano.shared(data_y, borrow=True)
    return shared_x, shared_y

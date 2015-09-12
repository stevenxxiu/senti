
import numpy as np
import theano
import theano.tensor as T
from keras.constraints import identity, maxnorm
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Graph
from keras.optimizers import Adadelta
from sklearn.base import BaseEstimator

__all__ = ['ConvNet']


def st(name, i):
    return '{}_{}'.format(name, i)


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
        self.classes_ = np.arange(hidden_units[1])

        self.y = T.ivector('y')

        model = Graph()
        model.add_input('input', dtype='int32')
        model.add_node(Embedding(
            word_vecs.shape[0], word_vecs.shape[1], weights=[word_vecs],
            W_constraint=(None if non_static else identity)
        ), 'embedding', 'input')
        for filter_h in filter_hs:
            model.add_node(Convolution1D(
                img_w, hidden_units[0], filter_h, activation=conv_non_linear
            ), st('conv', filter_h), 'embedding')
            model.add_node(MaxPooling1D(
                img_h - filter_h + 1, ignore_border=True
            ), st('maxpool', filter_h), st('conv', filter_h))
            model.add_node(Flatten(), st('flatten', filter_h), st('maxpool', filter_h))
        hidden_units[0] *= len(filter_hs)
        model.add_node(Dropout(
            dropout_rates[0]
        ), st('dropout', 0), inputs=list(st('flatten', filter_h) for filter_h in filter_hs))
        i = -1
        for i, (n_in, n_out, activation, dropout) in \
                enumerate(zip(hidden_units, hidden_units[1:-1], activations, dropout_rates[1:])):
            model.add_node(Dense(
                n_in, n_out, init='uniform', activation=activation, W_constraint=maxnorm(norm_lim)
            ), st('dense', i), st('dropout', i))
            model.add_node(Dropout(dropout), st('dropout', i + 1), st('dense', i))
        model.add_node(Dense(
            hidden_units[-2], hidden_units[-1], init='uniform', activation='softmax', W_constraint=maxnorm(norm_lim)
        ), 'output', st('dropout', i + 1), create_output=True)

        self.classifier = model
        self.classifier.errors = \
            lambda y: T.mean(T.neq(T.argmax(model.get_output(False), axis=1), y))
        self.classifier.negative_log_likelihood = \
            lambda y: -T.mean(T.log(model.get_output(False))[T.arange(y.shape[0]), y])
        self.classifier.dropout_negative_log_likelihood = \
            lambda y: -T.mean(T.log(model.get_output(True))[T.arange(y.shape[0]), y])

        dropout_cost = self.classifier.dropout_negative_log_likelihood(self.y)
        self.grad_updates = Adadelta(rho=lr_decay).get_updates(self.classifier.params, model.constraints, dropout_cost)

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

        # models
        index = T.lscalar()
        val_model = theano.function([index], self.classifier.errors(self.y), givens={
            self.classifier.inputs['input'].input: val_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            self.y: val_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })
        test_model = theano.function([index], self.classifier.errors(self.y), givens={
            self.classifier.inputs['input'].input: train_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            self.y: train_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })
        train_model = theano.function(
            [index], self.classifier.negative_log_likelihood(self.y), updates=self.grad_updates, givens={
                self.classifier.inputs['input'].input: train_set_x[index*self.batch_size:(index + 1)*self.batch_size],
                self.y: train_set_y[index*self.batch_size:(index + 1)*self.batch_size]
            }
        )

        # start training over mini-batches
        print('training cnn...')
        epoch = 0
        while epoch < self.n_epochs:
            epoch += 1
            if self.shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    train_model(minibatch_index)
            else:
                for minibatch_index in range(n_train_batches):
                    train_model(minibatch_index)
            train_losses = [test_model(i) for i in range(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_model(i) for i in range(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)
            print('epoch {}, train perf {} %, val perf {} %'.format(epoch, train_perf*100, val_perf*100))

    def predict_proba(self, X):
        # XXX
        pass


def shared_dataset(data_xy, borrow=True):
    '''
    Function that loads the dataset into shared variables. The reason we store our dataset in shared variables is to
    allow Theano to copy it into the GPU memory (when code is run on GPU). Since copying data into the GPU is slow,
    copying a minibatch everytime is needed (the default behaviour if the data is not in a shared variable) would lead
    to a large decrease in performance.
    '''
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype='int32'), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype='int32'), borrow=borrow)
    return shared_x, shared_y

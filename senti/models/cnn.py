
import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Layer

from senti.models.utils import *

__all__ = ['ConvNet']


class InputLayer(Layer):
    def __init__(self, input_):
        super().__init__()
        self.input = input_

    def get_input(self, train=False):
        return self.input


class ConvNet(BaseEstimator):
    def __init__(
        self, word_vecs, img_w, img_h, filter_hs, hidden_units, dropout_rates, conv_non_linear, activations, non_static,
        shuffle_batch, n_epochs, batch_size, lr_decay, sqr_norm_lim
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
        rng = np.random.RandomState(3435)
        filter_w = img_w
        feature_maps = hidden_units[0]
        filter_shapes = []
        pool_sizes = []
        for filter_h in filter_hs:
            filter_shapes.append((feature_maps, 1, filter_h, filter_w))
            pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.words = theano.shared(value=word_vecs, name='words')
        layer0_input = self.words[T.cast(self.x.flatten(), dtype='int32')] \
            .reshape((self.x.shape[0], 1, self.x.shape[1], self.words.shape[1]))
        self.conv_layers = []
        layer1_inputs = []
        for i in range(len(filter_hs)):
            filter_shape = filter_shapes[i]
            pool_size = pool_sizes[i]
            conv_layer = LeNetConvPoolLayer(
                rng, input=layer0_input, image_shape=(batch_size, 1, img_h, img_w), filter_shape=filter_shape,
                poolsize=pool_size, non_linear=conv_non_linear
            )
            layer1_input = conv_layer.output.flatten(2)
            self.conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        layer1_input = T.concatenate(layer1_inputs, 1)
        hidden_units[0] = feature_maps*len(filter_hs)

        model = Sequential()
        self._layer1 = InputLayer(layer1_input)
        model.add(self._layer1)
        model.add(Dropout(dropout_rates[0]))
        for n_in, n_out, activation, dropout in zip(hidden_units, hidden_units[1:-1], activations, dropout_rates[1:]):
            model.add(Dense(n_in, n_out, init='uniform', activation=activation))
            model.add(Dropout(dropout))
        model.add(Dense(hidden_units[-2], hidden_units[-1], init='uniform', activation='softmax'))
        self.classifier = model
        self.classifier.errors = \
            lambda y: T.mean(T.neq(T.argmax(model.get_output(False), axis=1), y))
        self.classifier.negative_log_likelihood = \
            lambda y: -T.mean(T.log(model.get_output(False))[T.arange(y.shape[0]), y])
        self.classifier.dropout_negative_log_likelihood = \
            lambda y: -T.mean(T.log(model.get_output(True))[T.arange(y.shape[0]), y])

        # define parameters of the model and update functions using adadelta
        params = self.classifier.params
        for conv_layer in self.conv_layers:
            params += conv_layer.params
        if non_static:
            # if word vectors are allowed to change, add them as model parameters
            params += [self.words]
        dropout_cost = self.classifier.dropout_negative_log_likelihood(self.y)
        self.grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

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
            self.x: val_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            self.y: val_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })
        test_model = theano.function([index], self.classifier.errors(self.y), givens={
            self.x: train_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            self.y: train_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })
        train_model = theano.function(
            [index], self.classifier.negative_log_likelihood(self.y), updates=self.grad_updates, givens={
                self.x: train_set_x[index*self.batch_size:(index + 1)*self.batch_size],
                self.y: train_set_y[index*self.batch_size:(index + 1)*self.batch_size]
            }
        )
        zero_vec_tensor = T.vector()
        zero_vec = np.zeros(self.img_w)
        set_zero = theano.function(
            [zero_vec_tensor], updates=[(self.words, T.set_subtensor(self.words[0, :], zero_vec_tensor))]
        )

        # start training over mini-batches
        print('training cnn...')
        epoch = 0
        while epoch < self.n_epochs:
            epoch += 1
            if self.shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    train_model(minibatch_index)
                    set_zero(zero_vec)
            else:
                for minibatch_index in range(n_train_batches):
                    train_model(minibatch_index)
                    set_zero(zero_vec)
            train_losses = [test_model(i) for i in range(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_model(i) for i in range(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)
            print('epoch {}, train perf {} %, val perf {} %'.format(epoch, train_perf*100, val_perf*100))

    def predict_proba(self, X):
        test_pred_layers = []
        num_docs = X.shape[0]
        test_layer0_input = self.words[T.cast(self.x.flatten(), dtype='int32')] \
            .reshape((num_docs, 1, self.img_h, self.words.shape[1]))
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, num_docs)
            test_pred_layers.append(test_layer0_output.flatten(2))
        self._layer1.input = T.concatenate(test_pred_layers, 1)
        test_model = theano.function([self.x], self.classifier.get_output(False))
        return test_model(X)


def shared_dataset(data_xy, borrow=True):
    '''
    Function that loads the dataset into shared variables. The reason we store our dataset in shared variables is to
    allow Theano to copy it into the GPU memory (when code is run on GPU). Since copying data into the GPU is slow,
    copying a minibatch everytime is needed (the default behaviour if the data is not in a shared variable) would lead
    to a large decrease in performance.
    '''
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator

from senti.models.utils import *

__all__ = ['ConvNet']


class ConvNet(BaseEstimator):
    def __init__(
        self, word_vecs, img_w, filter_hs, hidden_units, dropout_rate, shuffle_batch=True, n_epochs=25, batch_size=50,
        lr_decay=0.95, conv_non_linear='relu', activations=(Iden,), sqr_norm_lim=9, non_static=True
    ):
        '''
        Train a simple conv net
        img_h = sentence length (padded where necessary)
        img_w = word vector length (300 for word2vec)
        filter_hs = filter window sizes
        hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
        sqr_norm_lim = s^2 in the paper
        lr_decay = adadelta decay parameter
        '''
        self.word_vecs = word_vecs
        self.img_w = img_w
        self.filter_hs = filter_hs
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.shuffle_batch = shuffle_batch
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.conv_non_linear = conv_non_linear
        self.activations = activations
        self.sqr_norm_lim = sqr_norm_lim
        self.non_static = non_static

        # XXX temporary, separate out train & predict
        self.train_X = None
        self.train_y = None
        self.test_y = None

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
        return self

    def predict(self, X):
        # XXX separate out train & predict
        pass

    def predict_proba(self, X):
        # XXX separate out train & predict
        datasets = (np.hstack([self.train_X, self.train_y.reshape((-1, 1))]), np.hstack([X, self.test_y.reshape((-1, 1))]))
        rng = np.random.RandomState(3435)
        # XXX for now, make datasets the original
        img_h = len(datasets[0][0]) - 1
        filter_w = self.img_w
        feature_maps = self.hidden_units[0]
        filter_shapes = []
        pool_sizes = []
        for filter_h in self.filter_hs:
            filter_shapes.append((feature_maps, 1, filter_h, filter_w))
            pool_sizes.append((img_h - filter_h + 1, self.img_w - filter_w + 1))

        # define model architecture
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')
        Words = theano.shared(value=self.word_vecs, name="Words")
        zero_vec_tensor = T.vector()
        zero_vec = np.zeros(self.img_w)
        set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])
        layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
        conv_layers = []
        layer1_inputs = []
        for i in range(len(self.filter_hs)):
            filter_shape = filter_shapes[i]
            pool_size = pool_sizes[i]
            conv_layer = LeNetConvPoolLayer(
                rng, input=layer0_input, image_shape=(self.batch_size, 1, img_h, self.img_w), filter_shape=filter_shape,
                poolsize=pool_size, non_linear=self.conv_non_linear
            )
            layer1_input = conv_layer.output.flatten(2)
            conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        layer1_input = T.concatenate(layer1_inputs, 1)
        self.hidden_units[0] = feature_maps*len(self.filter_hs)
        classifier = MLPDropout(
            rng, input=layer1_input, layer_sizes=self.hidden_units, activations=self.activations,
            dropout_rates=self.dropout_rate
        )

        # define parameters of the model and update functions using adadelta
        params = classifier.params
        for conv_layer in conv_layers:
            params += conv_layer.params
        if self.non_static:
            # if word vectors are allowed to change, add them as model parameters
            params += [Words]
        cost = classifier.negative_log_likelihood(y)
        dropout_cost = classifier.dropout_negative_log_likelihood(y)
        grad_updates = sgd_updates_adadelta(params, dropout_cost, self.lr_decay, 1e-6, self.sqr_norm_lim)

        # shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
        # extra data (at random)
        np.random.seed(3435)
        if datasets[0].shape[0] % self.batch_size > 0:
            extra_data_num = self.batch_size - datasets[0].shape[0]%self.batch_size
            train_set = np.random.permutation(datasets[0])
            extra_data = train_set[:extra_data_num]
            new_data = np.append(datasets[0], extra_data,axis=0)
        else:
            new_data = datasets[0]
        new_data = np.random.permutation(new_data)
        n_batches = new_data.shape[0]//self.batch_size
        n_train_batches = int(np.round(n_batches*0.9))
        # divide train set into train/val sets
        test_set_x = datasets[1][:, :img_h]
        test_set_y = np.asarray(datasets[1][:, -1], "int32")
        train_set = new_data[:n_train_batches*self.batch_size, :]
        val_set = new_data[n_train_batches*self.batch_size:, :]
        train_set_x, train_set_y = shared_dataset((train_set[:, :img_h], train_set[:,-1]))
        val_set_x, val_set_y = shared_dataset((val_set[:, :img_h], val_set[:, -1]))
        n_val_batches = n_batches - n_train_batches
        val_model = theano.function([index], classifier.errors(y), givens={
            x: val_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            y: val_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })

        # compile theano functions to get train/val/test errors
        test_model = theano.function([index], classifier.errors(y), givens={
            x: train_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            y: train_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })
        train_model = theano.function([index], cost, updates=grad_updates, givens={
            x: train_set_x[index*self.batch_size:(index + 1)*self.batch_size],
            y: train_set_y[index*self.batch_size:(index + 1)*self.batch_size]
        })
        test_pred_layers = []
        test_size = test_set_x.shape[0]
        test_layer0_input = Words[T.cast(x.flatten(), dtype='int32')].reshape((test_size, 1, img_h, Words.shape[1]))
        for conv_layer in conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = classifier.predict(test_layer1_input)
        test_error = T.mean(T.neq(test_y_pred, y))
        test_model_all = theano.function([x,y], test_error)

        # start training over mini-batches
        print('... training')
        epoch = 0
        best_val_perf = 0
        val_perf = 0
        test_perf = 0
        cost_epoch = 0
        while epoch < self.n_epochs:
            epoch += 1
            if self.shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = train_model(minibatch_index)
                    set_zero(zero_vec)
            else:
                for minibatch_index in range(n_train_batches):
                    cost_epoch = train_model(minibatch_index)
                    set_zero(zero_vec)
            train_losses = [test_model(i) for i in range(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_model(i) for i in range(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)
            print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf*100., val_perf*100.))
            if val_perf >= best_val_perf:
                best_val_perf = val_perf
                test_loss = test_model_all(test_set_x,test_set_y)
                test_perf = 1 - test_loss
        return test_perf


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

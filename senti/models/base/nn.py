
import itertools
import logging

import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from senti.rand import get_rng
from senti.utils import log_time

__all__ = ['geometric_learning_rates', 'NNBase']


def geometric_learning_rates(init, ratio=None, repeat=None, n=0):
    learning_rate = init
    for i in range(n):
        yield from itertools.repeat([learning_rate], repeat)
        learning_rate *= ratio
    yield from itertools.repeat([learning_rate])


class EpochIterator:
    def __init__(self, gen_batches, args, epoch_size):
        self.gen_batches = gen_batches
        self.args = args
        self.epoch_size = epoch_size
        self.batch_iter = iter(())
        self.train_pass = 0

    def _iter_batches(self):
        i = 0
        while True:
            try:
                if self.epoch_size is not None and i >= self.epoch_size:
                    return
                yield next(self.batch_iter)
                i += 1
            except StopIteration:
                if self.epoch_size is None and i > 0:
                    return
                logging.info('training set pass {}'.format(self.train_pass + 1))
                self.batch_iter = self.gen_batches(*self.args)
                self.train_pass += 1

    def __iter__(self):
        while True:
            yield self._iter_batches()


class NNBase(BaseEstimator):
    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        self.args = args
        self.kwargs = kwargs
        self.classes_ = None
        self.network = None
        self.inputs = self.target = self.loss = self.probs = None
        self.updates, self.constraints = {}, {}
        self.update_params = []
        self.create_model(*args, **kwargs)
        for param, constraint in self.constraints.items():
            self.updates[param] = constraint(param, self.updates[param])

    def get_params(self, deep=True):
        return {'batch_size': self.batch_size, 'args': self.args, 'kwargs': self.kwargs}

    def create_model(self, *args, **kwargs):
        raise NotImplementedError

    def gen_batch(self, docs, y=None):
        raise NotImplementedError

    def gen_batches(self, docs, y=None):
        docs = list(zip(docs, y)) if y is not None else list(docs)
        if y is not None:
            get_rng().shuffle(docs)
        for i in range(0, len(docs), self.batch_size):
            cur_docs = docs[i:i + self.batch_size]
            if len(cur_docs) < self.batch_size:
                cur_docs.extend(docs[i] for i in get_rng().choice(len(docs), self.batch_size - len(cur_docs), False))
            yield self.gen_batch(*zip(*cur_docs)) if y is not None else self.gen_batch(cur_docs)

    @staticmethod
    def perf(epoch, train_res, dev_res, dev_y, average_classes):
        train_loss, train_acc = np.mean(train_res, axis=0)
        dev_acc = accuracy_score(dev_res, dev_y)
        dev_f1 = np.mean(precision_recall_fscore_support(dev_res, dev_y)[2][average_classes])
        logging.info('epoch {}, train loss {:.4f}, train acc {:.4f}, val acc {:.4f}, val f1 {:.4f}'.format(
            epoch + 1, train_loss, train_acc, dev_acc, dev_f1
        ))
        return dev_f1

    def fit(
        self, docs, y, dev_X, dev_y, average_classes, epoch_size=None, max_epochs=None,
        update_params_iter=itertools.repeat([]), save_best=True
    ):
        self.classes_ = unique_labels(dev_y)
        predictions = T.argmax(self.probs, axis=1)
        acc = T.mean(T.eq(predictions, self.target))
        train = theano.function(
            [*self.inputs, self.target, *self.update_params], [self.loss, acc], updates=self.updates
        )
        test = theano.function(self.inputs, predictions)
        with log_time('training...', 'training took {:.0f}s'):
            params = lasagne.layers.get_all_params(self.network)
            best_perf, best_params = None, None
            epoch_iter = EpochIterator(self.gen_batches, (docs, y), epoch_size//self.batch_size if epoch_size else None)
            for i, batches, update_params in zip(range(max_epochs), epoch_iter, update_params_iter):
                train_res = [train(*batch, *update_params) for batch in batches]
                dev_res = np.hstack(test(*batch[:-1]) for batch in self.gen_batches(dev_X))[:len(dev_y)]
                perf = self.perf(i, train_res, dev_res, dev_y, average_classes)
                if save_best and (best_perf is None or perf >= best_perf):
                    best_perf = perf
                    best_params = {param: param.get_value() for param in params}
            if save_best:
                for param, value in best_params.items():
                    param.set_value(value)

    def predict_proba(self, docs):
        predict = theano.function(self.inputs, self.probs)
        return np.vstack(predict(*batch[:-1]) for batch in self.gen_batches(docs))[:sum(1 for _ in docs)]

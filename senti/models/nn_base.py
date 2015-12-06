
import itertools

import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

__all__ = ['NNBase']


class EpochIterator:
    def __init__(self, gen_batches, args, epoch_len):
        self.gen_batches = gen_batches
        self.args = args
        self.epoch_len = epoch_len
        self.batch_iter = iter(())
        self.train_pass = 0

    def _iter_batches(self):
        for i in itertools.count(0):
            try:
                if self.epoch_len is not None and i >= self.epoch_len:
                    return
                yield next(self.batch_iter)
            except StopIteration:
                if self.epoch_len is None and i > 0:
                    return
                print('training set pass {}'.format(self.train_pass))
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
        self.inputs = self.target = None
        self.constraints = {}
        self.updates = self.loss = self.probs = None

    def create_model(self, *args, **kwargs):
        raise NotImplementedError

    def gen_batches(self, X, y):
        raise NotImplementedError

    @staticmethod
    def perf(epoch, train_res, dev_res, dev_y, average_classes):
        train_loss, train_acc = np.mean(train_res, axis=0)
        dev_acc = accuracy_score(dev_res, dev_y)
        dev_f1 = np.mean(precision_recall_fscore_support(dev_res, dev_y)[2][average_classes])
        print('epoch {}, train loss {:.4f}, train acc {:.4f}, val acc {:.4f}, val f1 {:.4f}'.format(
            epoch + 1, train_loss, train_acc, dev_acc, dev_f1
        ))
        return dev_f1

    def fit(self, docs, y, dev_X, dev_y, average_classes, epoch_len=None, max_epochs=None):
        if not self.network:
            self.create_model(*self.args, **self.kwargs)
            for param, constraint in self.constraints.items():
                self.updates[param] = constraint(param, self.updates[param])
        self.classes_ = unique_labels(dev_y)
        predictions = T.argmax(self.probs, axis=1)
        acc = T.mean(T.eq(predictions, self.target))
        train = theano.function(self.inputs + [self.target], [self.loss, acc], updates=self.updates)
        test = theano.function(self.inputs, predictions)
        print('training...')
        params = lasagne.layers.get_all_params(self.network)
        best_perf, best_params = None, None
        epoch_iter = EpochIterator(self.gen_batches, (docs, y), epoch_len//self.batch_size if epoch_len else None)
        for i, batches in zip(range(max_epochs), epoch_iter):
            train_res = [train(*batch) for batch in batches]
            dev_res = np.hstack(test(*data) for data in self.gen_batches(dev_X, None))[:len(dev_y)]
            perf = self.perf(i, train_res, dev_res, dev_y, average_classes)
            if best_perf is None or perf >= best_perf:
                best_perf = perf
                best_params = {param: param.get_value() for param in params}
        for param, value in best_params.items():
            param.set_value(value)

    def predict_proba(self, docs):
        predict = theano.function(self.inputs, self.probs)
        return np.vstack(predict(*data) for data in self.gen_batches(docs, None))[:sum(1 for _ in docs)]

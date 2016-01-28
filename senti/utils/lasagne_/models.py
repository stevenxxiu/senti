
import itertools
import logging

import numpy as np
import theano
from lasagne.layers import get_all_params
from sklearn.base import BaseEstimator

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
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.network = None
        self.inputs = self.update_params = []
        self.target = self.pred = self.loss = None
        self.updates, self.constraints = {}, {}
        self.metrics = {'train': [], 'val': []}
        self._train = self._test = None

    def compile(self):
        for param, constraint in self.constraints.items():
            self.updates[param] = constraint(param, self.updates[param])
        metric_outputs = [metric(self.target, self.pred, True) for metric in self.metrics['train']]
        self._train = theano.function(
            [*self.inputs, self.target, *self.update_params], [self.loss, *metric_outputs], updates=self.updates
        )
        self._test = theano.function(self.inputs, self.pred)

    def get_params(self, deep=True):
        return {
            'batch_size': self.batch_size, 'network': self.network, 'inputs': self.inputs,
            'update_params': self.update_params, 'target': self.target, 'loss': self.loss, 'updates': self.updates,
            'constraints': self.constraints, 'metrics': self.metrics
        }

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

    def perf(self, epoch, train_res, val_y=None, val_res=None):
        loss, *metrics = np.mean(train_res, axis=0)
        logs = ['epoch {}'.format(epoch + 1), 'train loss {:.4f}'.format(loss)]
        for metric, value in zip(self.metrics['train'], metrics):
            logs.append('train {} {:.4f}'.format(metric.__name__, value))
        res = None
        if val_y is not None:
            for metric in self.metrics['val']:
                res = metric(val_y, val_res)
                logs.append('val {} {:.4f}'.format(metric.__name__, res))
        logging.info(', '.join(logs))
        return res

    def fit(
        self, docs, y, max_epochs, epoch_size=None, val_docs=None, val_y=None, update_params_iter=itertools.repeat([]),
        save_best=True
    ):
        has_val = val_docs is not None
        with log_time('training...', 'training took {:.0f}s'):
            params = get_all_params(self.network)
            best_perf, best_params = None, None
            epoch_iter = EpochIterator(
                self.gen_batches, (docs, y), (epoch_size + self.batch_size - 1) // self.batch_size
                if epoch_size else None
            )
            for i, batches, update_params in zip(range(max_epochs), epoch_iter, update_params_iter):
                train_res = [self._train(*batch, *update_params) for batch in batches]
                val_res = np.concatenate(
                    [self._test(*batch[:-1]) for batch in self.gen_batches(val_docs)], axis=0
                )[:len(val_y)] if has_val else None
                perf = self.perf(i, train_res, val_y, val_res)
                if (has_val and save_best) and (best_perf is None or perf >= best_perf):
                    best_perf = perf
                    best_params = {param: param.get_value() for param in params}
            if has_val and save_best:
                for param, value in best_params.items():
                    param.set_value(value)

    def predict(self, docs):
        docs = list(docs)
        return np.vstack(self._test(*batch[:-1]) for batch in self.gen_batches(docs))[:len(docs)]

    def predict_proba(self, docs):
        docs = list(docs)
        return np.vstack(self._test(*batch[:-1]) for batch in self.gen_batches(docs))[:len(docs)]

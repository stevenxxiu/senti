
import itertools
import json
import sys

import numpy as np
from scipy import sparse

__all__ = ['Tee', 'PicklableSr', 'FieldExtractor', 'HeadSr', 'indexes_of', 'sparse_sum', 'vstack']


class Tee:
    def __init__(self, *args, **kwargs):
        self.file = open(*args, **kwargs)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


class PicklableSr:
    def __init__(self, sr):
        self.sr = sr
        self.name = sr.name
        self.encoding = sr.encoding

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['sr']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sr = open(self.name, encoding=self.encoding)


class FieldExtractor(PicklableSr):
    def __init__(self, sr, field):
        super().__init__(sr)
        self.field = field

    def __iter__(self):
        self.sr.seek(0)
        for line in self.sr:
            yield json.loads(line)[self.field]


class HeadSr(PicklableSr):
    def __init__(self, sr, n):
        super().__init__(sr)
        self.n = n

    def __iter__(self):
        self.sr.seek(0)
        for i, line in zip(range(self.n), self.sr):
            yield line


def indexes_of(x, y):
    '''
    Find the indexes of y in x.
    '''
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    return np.take(index, sorted_index, mode='clip')


def sparse_sum(X, axis):
    if axis == 0:
        n = X.shape[0]
        return sparse.csc_matrix((np.ones(n), (np.zeros(n), np.arange(n))))*X
    elif axis == 1:
        n = X.shape[1]
        return X*sparse.csc_matrix((np.ones(n), (np.zeros(n), np.arange(n))))


def vstack(Xs):
    Xs = iter(Xs)
    X = next(Xs)
    Xs = itertools.chain([X], Xs)
    return sparse.vstack(Xs) if X.ndim == 2 else np.vstack(Xs)

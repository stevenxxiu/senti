
import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.numpy_ import sparse_sum, vstack

__all__ = ['Map', 'MapTokens', 'Index', 'Count', 'Proportion', 'Clip']


class Map(BaseEstimator):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    @reiterable
    def transform(self, docs):
        for doc in docs:
            yield self.func(doc)


class MapTokens(BaseEstimator):
    def __init__(self, func):
        self.func = func

    def fit(self, docs, y=None):
        return self

    @reiterable
    def transform(self, docs):
        for doc in docs:
            yield list(map(self.func, doc))


class Index(BaseEstimator):
    def __init__(self, i):
        self.i = i

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        return vstack(doc[self.i] for doc in docs)


class Count(BaseEstimator):
    def fit(self, docs, y=None):
        return self

    @staticmethod
    def transform(docs):
        return vstack(sparse_sum(doc, axis=0) for doc in docs)


class Proportion(BaseEstimator):
    def fit(self, docs, y=None):
        return self

    @staticmethod
    def transform(docs):
        return vstack(sparse_sum(doc, axis=0)/doc.shape[0] for doc in docs)


class Clip(BaseEstimator):
    def __init__(self, max_len):
        self.max_len = max_len

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        return vstack(np.hstack([
                doc[:self.max_len], np.zeros(max(0, self.max_len - len(doc)), dtype=doc.dtype)
        ]) for doc in docs)

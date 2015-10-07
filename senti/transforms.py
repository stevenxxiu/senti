
import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable, sparse_sum, vstack

__all__ = ['FixedTransformWrapper', 'Map', 'MapTokens', 'Index', 'Count', 'Proportion', 'Clip']


class FixedTransformWrapper(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self._corporas = []
        self._corporas_start = []
        self._corporas_end = []

    @reiterable
    def _chain_docs(self, corporas):
        self._corporas = corporas
        pos = 0
        for corpora in corporas:
            self._corporas_start.append(pos)
            i = -1
            for i, doc in enumerate(corpora):
                yield doc
            pos += i + 1
            self._corporas_end.append(pos)

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def fit(self, corporas, y=None):
        self.estimator.fit(self._chain_docs(corporas), y)
        return self

    def transform(self, docs):
        try:
            i = self._corporas.index(docs)
        except IndexError:
            raise ValueError('docs were not fitted')
        return self.estimator.transform(docs)[self._corporas_start[i]:self._corporas_end[i]]


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
        return vstack(np.hstack([doc[:self.max_len], np.zeros(self.max_len - len(doc))]) for doc in docs)

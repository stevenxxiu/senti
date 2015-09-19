
from scipy import sparse
from sklearn.base import BaseEstimator

from senti.base import ReiterableMixin
from senti.utils import sparse_sum, vstack

__all__ = ['Map', 'Index', 'Count', 'Proportion', 'Clip']


class Map(BaseEstimator, ReiterableMixin):
    def __init__(self, funcs):
        self.funcs = funcs

    def fit(self, X, y=None):
        return self

    def _transform(self, docs):
        for doc in docs:
            for func in self.funcs[::-1]:
                doc = func(doc)
            yield doc


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
        return vstack(sparse.coo_matrix(sparse_sum(doc, axis=0)) for doc in docs)


class Proportion(BaseEstimator):
    def fit(self, docs, y=None):
        return self

    @staticmethod
    def transform(docs):
        return vstack(sparse.coo_matrix(sparse_sum(doc, axis=0))/doc.shape[0] for doc in docs)


class Clip(BaseEstimator):
    def __init__(self, max_len):
        self.max_len = max_len

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        return vstack(doc[:self.max_len] for doc in docs)

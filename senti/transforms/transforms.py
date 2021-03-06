
import itertools

from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.numpy_ import sparse_sum, vstack
from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['Map', 'MapTokens', 'Zip', 'Repeat', 'Index', 'Count', 'Proportion']


class Map(BaseEstimator, EmptyFitMixin):
    def __init__(self, func):
        self.func = func

    @reiterable
    def transform(self, docs):
        for doc in docs:
            yield self.func(doc)


class MapTokens(BaseEstimator, EmptyFitMixin):
    def __init__(self, func):
        self.func = func

    @reiterable
    def transform(self, docs):
        for doc in docs:
            yield list(map(self.func, doc))


class Zip(BaseEstimator, EmptyFitMixin):
    def __init__(self, *estimators):
        self.estimators = estimators

    @reiterable
    def transform(self, docs):
        yield from itertools.zip_longest(*(estimator.transform(docs) for estimator in self.estimators))


class Repeat(BaseEstimator, EmptyFitMixin):
    def __init__(self, n):
        self.n = n

    @reiterable
    def transform(self, docs):
        for doc in docs:
            yield doc * self.n


class Index(BaseEstimator, EmptyFitMixin):
    def __init__(self, i):
        self.i = i

    def transform(self, docs):
        return vstack(doc[self.i] for doc in docs)


class Count(BaseEstimator, EmptyFitMixin):
    @staticmethod
    def transform(docs):
        return vstack(sparse_sum(doc, axis=0) for doc in docs)


class Proportion(BaseEstimator, EmptyFitMixin):
    @staticmethod
    def transform(docs):
        return vstack(sparse_sum(doc, axis=0) / doc.shape[0] for doc in docs)


from sklearn.base import BaseEstimator

__all__ = ['MapTransform']


class ReusableMap:
    def __init__(self, docs, funcs):
        self.docs = docs
        self.funcs = funcs

    def __eq__(self, other):
        return self.docs == other.docs and self.funcs == other.funcs

    def __iter__(self):
        for doc in self.docs:
            for func in self.funcs[::-1]:
                doc = func(doc)
            yield doc


class MapTransform(BaseEstimator):
    def __init__(self, funcs):
        self.funcs = funcs

    def fit(self, X, y=None):
        return self

    def transform(self, docs):
        return ReusableMap(docs, self.funcs)


from sklearn.base import BaseEstimator

from senti.transforms.base import ReiterableMixin

__all__ = ['MapTransform']


class MapTransform(BaseEstimator, ReiterableMixin):
    def __init__(self, funcs):
        self.funcs = funcs

    def fit(self, X, y=None):
        return self

    def _transform(self, docs):
        for doc in docs:
            for func in self.funcs[::-1]:
                doc = func(doc)
            yield doc

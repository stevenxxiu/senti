
import numpy as np
from sklearn.base import BaseEstimator

from senti.transforms.base import ReiterableMixin

__all__ = ['Map', 'Clip']


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


class Clip(BaseEstimator):
    def __init__(self, max_len):
        self.max_len = max_len

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        vecs = []
        for doc in docs:
            vec = np.zeros(self.max_len)
            vec[:min(len(doc), self.max_len)] = doc[:self.max_len]
            vecs.append(vec)
        return np.vstack(vecs)

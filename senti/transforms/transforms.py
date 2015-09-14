
import numpy as np
from sklearn.base import BaseEstimator

from senti.transforms.base import ReiterableMixin

__all__ = ['MapTransform', 'ClipPad']


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


class ClipPad(BaseEstimator):
    def __init__(self, pad, max_len):
        self.pad = pad
        self.max_len = max_len

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        vecs = []
        for doc in docs:
            vec = np.zeros(self.max_len + self.pad*2, dtype='int32')
            doc = doc[:self.max_len]
            vec[self.pad:self.pad + len(doc)] = doc
            vecs.append(vec)
        return np.vstack(vecs)

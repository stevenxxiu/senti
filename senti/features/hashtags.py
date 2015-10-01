
import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable

__all__ = ['HashTags']


class HashTags(BaseEstimator):
    def fit(self, docs, y=None):
        return self

    @staticmethod
    @reiterable
    def transform(docs):
        for doc in docs:
            yield np.fromiter((word.startswith('#') for word in doc), dtype='int32')

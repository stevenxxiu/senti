
import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['HashTags']


class HashTags(BaseEstimator, EmptyFitMixin):
    @staticmethod
    @reiterable
    def transform(docs):
        for doc in docs:
            yield np.fromiter((word.startswith('#') for word in doc), dtype='int32')


import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['AllCaps']


class AllCaps(BaseEstimator, EmptyFitMixin):
    '''
    Fully capitalised letters.
    '''

    @staticmethod
    @reiterable
    def transform(docs):
        for doc in docs:
            yield np.fromiter((word.isupper() for word in doc), dtype='int32')

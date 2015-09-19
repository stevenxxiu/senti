
import numpy as np
from sklearn.base import BaseEstimator

from senti.base import ReiterableMixin

__all__ = ['AllCaps']


class AllCaps(BaseEstimator, ReiterableMixin):
    '''
    Fully capitalised letters.
    '''

    def fit(self, docs, y=None):
        return self

    @staticmethod
    def _transform(docs):
        for doc in docs:
            yield np.fromiter((word.isupper() for word in doc), dtype='int32')

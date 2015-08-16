
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['AllCaps']


class AllCaps(BaseEstimator):
    '''
    Proportion of tokens with fully capitalised letters.
    '''

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        rows = []
        for doc in docs:
            rows.append(sum(1 for token in doc if token.isupper())/len(doc))
        return np.vstack(rows)

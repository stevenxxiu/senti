
import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable

__all__ = ['Punctuations']


class Punctuations(BaseEstimator):
    '''
    Proportion of words with punctuation marks.
    '''

    def fit(self, docs, y=None):
        return self

    @staticmethod
    @reiterable
    def transform(docs):
        for doc in docs:
            charsets = tuple(frozenset(word) for word in doc if word)
            yield np.hstack([
                np.fromiter((chars == {'!'} for chars in charsets), dtype='bool'),
                np.fromiter((chars == {'!'} for chars in charsets), dtype='bool'),
                np.fromiter((chars == {'!', '?'} for chars in charsets), dtype='bool'),
            ])

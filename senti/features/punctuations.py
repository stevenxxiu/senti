
import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['Punctuations']


class Punctuations(BaseEstimator, EmptyFitMixin):
    '''
    Proportion of words with punctuation marks.
    '''

    @staticmethod
    @reiterable
    def transform(docs):
        for doc in docs:
            charsets = tuple(frozenset(word) for word in doc if word)
            yield np.hstack([
                np.fromiter((chars == {'!'} for chars in charsets), dtype='bool'),
                np.fromiter((chars == {'?'} for chars in charsets), dtype='bool'),
                np.fromiter((chars == {'!', '?'} for chars in charsets), dtype='bool'),
            ])

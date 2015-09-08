
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Punctuations']


class Punctuations(BaseEstimator):
    '''
    Proportion of tokens with punctuation marks.
    '''

    def fit(self, docs, y=None):
        return self

    @staticmethod
    def transform(docs):
        vecs = []
        for doc in docs:
            charsets = tuple(frozenset(token) for token in doc if token)
            vec = np.array([
                sum(chars == {'!'} for chars in charsets)/len(doc),
                sum(chars == {'?'} for chars in charsets)/len(doc),
                sum(chars == {'!', '?'} for chars in charsets)/len(doc),
                int(bool(charsets[-1] <= {'!', '?'}))
            ])
            vecs.append(vec)
        return np.vstack(vecs)

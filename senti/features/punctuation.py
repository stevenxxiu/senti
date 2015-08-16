
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Punctuation']


class Punctuation(BaseEstimator):
    '''
    Proportion of tokens with punctuation marks.
    '''

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, docs, y):
        return self

    def transform(self, docs):
        vecs = []
        for doc in docs:
            tokens = self.tokenizer(doc)
            charsets = tuple(frozenset(t) for t in tokens if t)
            vec = np.array([
                sum(chars == {'!'} for chars in charsets),
                sum(chars == {'?'} for chars in charsets),
                sum(chars == {'!', '?'} for chars in charsets),
                int(bool(charsets[-1] <= {'!', '?'}))
            ])/len(tokens)
            vecs.append(vec)
        return np.vstack(vecs)

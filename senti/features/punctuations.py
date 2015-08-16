
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Punctuations']


class Punctuations(BaseEstimator):
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
                sum(chars == {'!'} for chars in charsets)/len(tokens),
                sum(chars == {'?'} for chars in charsets)/len(tokens),
                sum(chars == {'!', '?'} for chars in charsets)/len(tokens),
                int(bool(charsets[-1] <= {'!', '?'}))
            ])
            vecs.append(vec)
        return np.vstack(vecs)

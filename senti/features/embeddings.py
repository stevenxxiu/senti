
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable

__all__ = ['Embeddings']


class Embeddings(BaseEstimator):
    def __init__(self, embeddings, rand=None, include_zero=True, min_df=1):
        self.embeddings = embeddings
        self.rand = rand
        if self.rand is None:
            self.X = embeddings.X
            self.vocab = embeddings.vocab
        else:
            self.X = np.empty((0, embeddings.X.shape[1]), dtype='float32')
            self.vocab = {}
        if include_zero:
            self.X = np.vstack([np.zeros(self.X.shape[1], dtype='float32'), self.X])
            self.vocab = dict((word, i + 1) for word, i in self.vocab.items())
        self.include_zero = include_zero
        self.min_df = min_df

    def fit(self, docs, y=None):
        if self.rand is None:
            return self
        dfs = Counter()
        for doc in docs:
            for word in doc:
                dfs[word] += 1
        vecs = []
        for word, df in sorted(dfs.items()):
            if word not in self.vocab and df >= self.min_df:
                self.vocab[word] = self.X.shape[0] + len(vecs)
                if self.embeddings and word in self.embeddings.vocab:
                    vecs.append(self.embeddings.X[self.embeddings.vocab[word]])
                else:
                    vecs.append(self.rand(self.X.shape[1]).astype('float32'))
        self.X = np.vstack([self.X] + vecs)
        return self

    @reiterable
    def transform(self, docs):
        for doc in docs:
            if self.include_zero:
                indexes = (self.vocab.get(word, 0) for word in doc)
            else:
                indexes = (self.vocab[word] for word in doc if word in self.vocab)
            yield np.fromiter(indexes, dtype='int32')

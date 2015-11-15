
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable

__all__ = ['EmbeddingConstructor']


class EmbeddingConstructor(BaseEstimator):
    def __init__(self, embeddings, rand, include_zero=True, min_df=1):
        self.embeddings = embeddings
        self.rand = rand
        self.X = np.zeros((int(include_zero), embeddings.X.shape[1]))
        self.vocab = {}
        self.include_zero = include_zero
        self.min_df = min_df

    def fit(self, docs, y=None):
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
                    vecs.append(self.rand(self.X.shape[1]))
        self.X = np.vstack([self.X] + vecs)
        return self

    @reiterable
    def transform(self, docs):
        for doc in docs:
            yield np.fromiter((self.vocab.get(word, 0) for word in doc), dtype='int32')

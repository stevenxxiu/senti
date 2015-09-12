
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Index']


class Index(BaseEstimator):
    def __init__(self, rand_vecs, embeddings=None, include_zero=True, min_df=1):
        self.rand_vecs = rand_vecs
        if embeddings:
            self.word_to_index = embeddings.word_to_index.copy()
            self.X = embeddings.X
            if include_zero and 0 in self.word_to_index.values():
                for word in self.word_to_index:
                    self.word_to_index[word] += 1
                self.X = np.vstack([np.zeros(embeddings.X.shape[1]), self.X])
        else:
            self.word_to_index = {}
            self.X = np.zeros_like(rand_vecs(1))
        self.include_zero = include_zero
        self.min_df = min_df

    def fit(self, docs, y=None):
        dfs = Counter()
        for doc in docs:
            for word in doc:
                dfs[word] += 1
        for word, df in dfs.items():
            if word not in self.word_to_index and df >= self.min_df:
                self.word_to_index[word] = self.include_zero + len(self.word_to_index)
        self.X = np.vstack([self.X, self.rand_vecs(self.include_zero + len(self.word_to_index) - self.X.shape[0])])
        return self

    def transform(self, docs):
        for doc in docs:
            indexes = []
            for word in doc:
                indexes.append(self.word_to_index.get(word, 0))
            yield indexes

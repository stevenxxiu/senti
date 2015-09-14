
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Index']


class Index(BaseEstimator):
    def __init__(self, rand_vec, embeddings=None, include_zero=True, min_df=1):
        self.rand_vec = rand_vec
        self.embeddings = embeddings
        self.X = np.zeros((int(include_zero), rand_vec().shape[0]))
        self.word_to_index = {}
        self.include_zero = include_zero
        self.min_df = min_df

    def fit(self, docs, y=None):
        dfs = Counter()
        for doc in docs:
            for word in doc:
                dfs[word] += 1
        vecs = []
        for word, df in dfs.items():
            if word not in self.word_to_index and df >= self.min_df:
                self.word_to_index[word] = self.X.shape[0] + len(vecs)
                if self.embeddings and word in self.embeddings.word_to_index:
                    vecs.append(self.embeddings.X[self.embeddings.word_to_index[word]])
                else:
                    vecs.append(self.rand_vec())
        self.X = np.vstack([self.X] + vecs)
        return self

    def transform(self, docs):
        for doc in docs:
            indexes = []
            for word in doc:
                indexes.append(self.word_to_index.get(word, 0))
            yield indexes

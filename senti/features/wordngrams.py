
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator

__all__ = ['WordNGrams']


class WordNGrams(BaseEstimator):
    def __init__(self, tokenizer, n_min=2, n_max=4, exclude=True):
        self.tokenizer = tokenizer
        self.n_min = n_min
        self.n_max = n_max
        self.exclude = exclude
        self.ngrams = {}
        self.counts = None

    def _iter_ngrams(self, doc):
        tokens = self.tokenizer(doc)
        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tokens[i:i + n]
                yield tuple(ngram)
                if self.exclude:
                    for j in range(1, n - 1):
                        excluded = list(ngram)
                        excluded[j] = None
                        yield tuple(excluded)

    def fit(self, docs, y):
        ngrams = Counter()
        for doc in docs:
            for ngram in self._iter_ngrams(doc):
                ngrams[ngram] += 1
        counts = []
        for i, (ngram, count) in enumerate(ngrams.items()):
            self.ngrams[ngram] = i
            counts.append(count)
        self.counts = np.array(counts)
        return self

    def transform(self, docs):
        data, row, col = [], [], []
        i = -1
        for i, doc in enumerate(docs):
            for ngram in self._iter_ngrams(doc):
                if ngram in self.ngrams:
                    row.append(i)
                    col.append(self.ngrams[ngram])
                    data.append(self.counts[col[-1]])
        return sparse.coo_matrix((data, (row, col)), shape=(i + 1, self.counts.shape[0]))

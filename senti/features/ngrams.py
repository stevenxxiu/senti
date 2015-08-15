
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator

__all__ = ['CharNGrams', 'WordNGrams']


class NGramsBase(BaseEstimator):
    def __init__(self):
        self.ngrams = {}
        self.counts = None

    def _iter_ngrams(self, doc):
        raise NotImplementedError

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


class WordNGrams(NGramsBase):
    def __init__(self, tokenizer, n_min=2, n_max=4, exclude=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_min = n_min
        self.n_max = n_max
        self.exclude = exclude

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


class CharNGrams(NGramsBase):
    def __init__(self, preprocessor, tokenizer, n_min=3, n_max=5, tokens_only=False, exclude=True):
        super().__init__()
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.n_min = n_min
        self.n_max = n_max
        self.tokens_only = tokens_only
        self.exclude = exclude

    def _iter_ngrams(self, doc):
        doc = self.preprocessor(doc)
        tokens = self.tokenizer(doc) if self.tokens_only else (doc,)
        for token in tokens:
            for n in range(self.n_min, self.n_max + 1):
                for i in range(len(token) - n + 1):
                    ngram = token[i:i + n]
                    yield tuple(ngram)

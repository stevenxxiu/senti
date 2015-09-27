
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator

from senti.base import ReiterableMixin

__all__ = ['CharNGrams', 'WordNGrams']


class NGramsBase(BaseEstimator, ReiterableMixin):
    def __init__(self):
        self.ngram_to_index = {}

    def _iter_ngrams(self, doc):
        raise NotImplementedError

    def fit(self, docs, y=None):
        for doc in docs:
            for ngram in self._iter_ngrams(doc):
                if ngram not in self.ngram_to_index:
                    self.ngram_to_index[ngram] = len(self.ngram_to_index)
        return self

    def _transform(self, docs):
        for doc in docs:
            # include 0 rows so the shape is right
            indices, indptr = [], [0]
            i = -1
            for i, ngram in enumerate(self._iter_ngrams(doc)):
                if ngram in self.ngram_to_index:
                    indices.append(self.ngram_to_index[ngram])
                indptr.append(len(indices))
            yield sparse.csr_matrix(
                (np.ones(len(indices)), indices, indptr), shape=(i + 1, len(self.ngram_to_index)), dtype='float32'
            )


class WordNGrams(NGramsBase):
    def __init__(self, n, exclude=True):
        super().__init__()
        self.n = n
        self.exclude = exclude

    def _iter_ngrams(self, doc):
        for i in range(len(doc) - self.n + 1):
            ngram = doc[i:i + self.n]
            yield tuple(ngram)
            if self.exclude:
                for j in range(1, self.n - 1):
                    excluded = list(ngram)
                    excluded[j] = None
                    yield tuple(excluded)


class CharNGrams(NGramsBase):
    def __init__(self, n, words_only=False):
        super().__init__()
        self.n = n
        self.words_only = words_only

    def _iter_ngrams(self, doc):
        if not self.words_only:
            doc = [' '.join(doc)]
        for word in doc:
            for i in range(len(word) - self.n + 1):
                ngram = word[i:i + self.n]
                yield tuple(ngram)

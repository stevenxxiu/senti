
from collections import Counter

import numpy as np

from senti.transform import PersistentTransform

__all__ = ['WordNGrams']


class WordNGrams(PersistentTransform):
    def __init__(self, nmin=2, nmax=4, expand=True):
        super().__init__('wordngrams', reuse_options=(nmin, nmax, expand))
        self.nmin = nmin
        self.nmax = nmax
        self.expand = expand
        self.counts = Counter()

    @PersistentTransform.persist
    def fit(self, train_sr):
        for obj in train_sr:
            tokens = obj['tokens']
            for n in range(self.nmin, self.nmax + 1):
                for i in range(len(tokens) - n):
                    ngram = tokens[i:i + n]
                    self.counts[str(ngram)] += 1
                    if self.expand:
                        for j in range(1, n - 1):
                            wild = list(ngram)
                            wild[j] = None
                            self.counts[str(ngram)] += 1
        return self

    @PersistentTransform.persist
    def transform(self, dev_sr):
        pass

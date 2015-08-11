
from collections import Counter

import numpy as np

__all__ = ['WordNGrams']


class WordNGrams:
    def __init__(self, nmin=2, nmax=4, expand=True):
        self.nmin = nmin
        self.nmax = nmax
        self.expand = expand
        self.counts = Counter()

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

    def transform(self, dev_sr):
        pass

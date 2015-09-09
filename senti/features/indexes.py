
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['index_words', 'IndexClipped']


def index_words(docs, min_index=0, min_df=1):
    word_to_index = {}
    dfs = Counter()
    for doc in docs:
        for word in doc:
            dfs[word] += 1
            if word not in word_to_index:
                word_to_index[word] = min_index + len(word_to_index)
    for word, df in dfs.items():
        if df < min_df:
            del word_to_index[word]
    return word_to_index


class IndexClipped(BaseEstimator):
    def __init__(self, word_to_index, pad, max_len):
        self.word_to_index = word_to_index
        self.pad = pad
        self.max_len = max_len

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        vecs = []
        for doc in docs:
            vec = np.zeros(self.max_len + self.pad*2)
            for i, word in zip(range(self.max_len), doc):
                if word in self.word_to_index:
                    vec[self.pad + i] = self.word_to_index[word]
            vecs.append(vec)
        return np.vstack(vecs)

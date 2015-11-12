
from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels

__all__ = ['Word2VecBayes']


class Word2VecBayes(BaseEstimator):
    def __init__(self, model):
        self.joint_model = model
        self.classes_ = None
        self.freqs = None
        self.models = None

    def fit(self, docs, y):
        self.joint_model.build_vocab(docs)
        self.classes_ = unique_labels(y)
        self.freqs = Counter(y)
        self.models = dict((c, deepcopy(self.joint_model)) for c in self.freqs)
        for c in self.models:
            cur_docs = list(doc for doc, c_ in zip(docs, y) if c_ == c)
            np.random.shuffle(cur_docs)
            cur_model = self.models[c]
            cur_model.train(cur_docs)
            cur_model.min_alpha = cur_model.alpha
            for epoch in range(20):
                np.random.shuffle(cur_docs)
                cur_model.train(cur_docs)
                cur_model.alpha *= 0.9
                cur_model.min_alpha = cur_model.alpha

    def predict_proba(self, docs):
        scores = np.hstack(self.models[c].score(docs).reshape(-1, 1) for c in self.freqs)
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


import logging
from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator

from senti.rand import get_rng

__all__ = ['Word2VecBayes']


class Word2VecBayes(BaseEstimator):
    def __init__(self, model, class_priors=True):
        self.joint_model = model
        self.class_priors = class_priors
        self.class_scores = None
        self.models = None

    def fit(self, docs, y):
        self.joint_model.build_vocab(docs)
        freqs = Counter(y)
        classes = sorted(freqs.keys())
        self.class_scores = np.log([freqs[c] for c in classes])
        self.models = [deepcopy(self.joint_model) for _ in classes]
        for class_, model in zip(classes, self.models):
            cur_docs = [doc for doc, c in zip(docs, y) if c == class_]
            for epoch in range(20):
                logging.info('epoch {}'.format(epoch + 1))
                get_rng().shuffle(cur_docs)
                model.train(cur_docs)
                model.alpha *= 0.9
                model.min_alpha = model.alpha

    def predict_proba(self, docs):
        scores = np.hstack(cur_model.score(docs).reshape(-1, 1) for cur_model in self.models)
        if self.class_priors:
            scores += self.class_scores
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

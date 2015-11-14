
import json

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels

__all__ = ['ExternalModel']


class ExternalModel(BaseEstimator):
    def __init__(self, docs_to_path):
        self.docs_to_path = docs_to_path
        self.classes_ = None

    def fit(self, docs, y=None):
        self.classes_ = unique_labels(y)
        return self

    def predict_proba(self, docs):
        with open(self.docs_to_path[docs]) as sr:
            probs = []
            for line in sr:
                obj = json.loads(line)
                probs.append(list(zip(*obj['probs']))[1])
            return np.vstack(probs)

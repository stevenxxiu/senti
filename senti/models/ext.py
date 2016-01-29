
import json

import numpy as np
from sklearn.base import BaseEstimator

from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['ExternalModel']


class ExternalModel(BaseEstimator, EmptyFitMixin):
    def __init__(self, docs_to_path):
        self.docs_to_path = docs_to_path

    def predict(self, docs):
        with open(self.docs_to_path[docs]) as sr:
            res = []
            for line in sr:
                res.append(json.loads(line)['label'])
            return res

    def predict_proba(self, docs):
        with open(self.docs_to_path[docs]) as sr:
            probs = []
            for line in sr:
                obj = json.loads(line)
                probs.append(list(zip(*obj['probs']))[1])
            return np.vstack(probs)

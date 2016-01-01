
import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['IntroduceTypos']


class IntroduceTypos(BaseEstimator, EmptyFitMixin):
    def __init__(self, alphabet, p=0.3):
        self.alphabet = alphabet
        self.p = p

    @reiterable
    def transform(self, docs):
        for doc in docs:
            res = ''
            is_ = np.random.choice(len(doc), min(len(doc), np.random.geometric(self.p) - 1), replace=False)
            prev_i = -1
            for i in is_:
                # delete, insert, substitute
                op = np.random.choice(3)
                if op == 0:
                    res += doc[prev_i + 1:i]
                elif op == 1:
                    res += doc[prev_i + 1:i + 1] + self.alphabet[np.random.choice(len(self.alphabet))]
                else:
                    res += doc[prev_i + 1:i] + self.alphabet[np.random.choice(len(self.alphabet))]
                prev_i = i
            res += doc[prev_i + 1:]
            yield res

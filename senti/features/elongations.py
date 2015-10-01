
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable

__all__ = ['Elongations']


class Elongations(BaseEstimator):
    '''
    Elongated vowels. Other possible constraints: letters, none.
    '''

    @staticmethod
    def is_vowel(c):
        return c in 'aeiou'

    @staticmethod
    def is_letter(c):
        return unicodedata.category(c)[0] == 'L'

    @staticmethod
    def is_elongated(word, condition=None):
        if len(word) >= 3:
            condition = condition or (lambda x: True)
            prevprev, prev = word[:2]
            for i in range(2, len(word)):
                char = word[i]
                if condition(char) and char == prev and char == prevprev:
                    return True
                prevprev, prev = prev, char
        return False

    def fit(self, docs, y=None):
        return self

    @reiterable
    def transform(self, docs):
        for doc in docs:
            yield np.fromiter((self.is_elongated(word) for word in doc), dtype='int32')

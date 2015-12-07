
import re

import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['NegationAppend', 'NegationCount']

# Note that Potts didn't included the comma in the definition, but his examples assume it terminates a context.
TERMINATORS = frozenset('.:;!?,')
NEGATION_RE = re.compile(r'''(?:^(?:
    never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|
    isnt|arent|aint
)$)|n't''', re.X | re.I)


class NegationBase(BaseEstimator):
    @staticmethod
    def is_negation(word):
        return bool(NEGATION_RE.search(word))

    @staticmethod
    def is_terminator(word):
        return word and TERMINATORS.issuperset(word)

    def find_negations(self, doc):
        negated = False
        for word in doc:
            if negated and self.is_terminator(word):
                negated = False
            yield negated
            if not negated and self.is_negation(word):
                negated = True


class NegationAppend(NegationBase, EmptyFitMixin):
    '''
    Appends '_NEG' to words in negated contexts.
    '''

    @reiterable
    def transform(self, docs):
        for doc in docs:
            for i, negated in enumerate(self.find_negations(doc)):
                if negated:
                    doc[i] += '_NEG'
            yield doc


class NegationCount(NegationBase, EmptyFitMixin):
    '''
    Counts the # of negation contexts.
    '''

    def transform(self, docs):
        vecs = []
        for doc in docs:
            c = 0
            prev_negated = False
            for i, negated in enumerate(self.find_negations(doc)):
                if negated != prev_negated:
                    c += 1
                prev_negated = negated
            vecs.append(c)
        return np.vstack(vecs)


import re

from sklearn.base import BaseEstimator

from senti.transforms.base import ReiterableMixin

__all__ = ['Negations']

# Note that Potts didn't included the comma in the definition, but his examples assume it terminates a context.
TERMINATORS = frozenset('.:;!?,')
NEGATION_RE = re.compile(r'''(?:
    ^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$
)|n't''', re.X | re.I | re.U)


class Negations(BaseEstimator, ReiterableMixin):
    '''
    Appends '_NEG' to tokens in negated contexts.
    '''

    @staticmethod
    def is_negation(token):
        return bool(NEGATION_RE.search(token))

    @staticmethod
    def is_terminator(token):
        return token and TERMINATORS.issuperset(token)

    def fit(self, docs, y=None):
        return self

    def _transform(self, docs):
        for doc in docs:
            negated = False
            for i, token in enumerate(doc):
                if negated and self.is_terminator(token):
                    negated = False
                if negated:
                    doc[i] = 'NEG_' + token
                if not negated and self.is_negation(token):
                    negated = True
            yield doc

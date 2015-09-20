import re

from sklearn.base import BaseEstimator

from senti.base import ReiterableMixin

__all__ = ['Negations']

# Note that Potts didn't included the comma in the definition, but his examples assume it terminates a context.
TERMINATORS = frozenset('.:;!?,')
NEGATION_RE = re.compile(r'''(?:^(?:
    never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|
    isnt|arent|aint
)$)|n't''', re.X | re.I | re.U)


class Negations(BaseEstimator, ReiterableMixin):
    '''
    Prepends 'neg_' to words in negated contexts.
    '''

    @staticmethod
    def is_negation(word):
        return bool(NEGATION_RE.search(word))

    @staticmethod
    def is_terminator(word):
        return word and TERMINATORS.issuperset(word)

    def fit(self, docs, y=None):
        return self

    def _transform(self, docs):
        for doc in docs:
            negated = False
            for i, word in enumerate(doc):
                if negated and self.is_terminator(word):
                    negated = False
                if negated:
                    doc[i] = 'neg_' + word
                if not negated and self.is_negation(word):
                    negated = True
            yield doc

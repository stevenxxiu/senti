
import re

import numpy as np
from sklearn.base import BaseEstimator

from senti.utils import reiterable

__all__ = ['Emoticons', 'EmoticonType', 'emoticon_re']

# Adapted from Christopher Potts' emoticon recognising tokeniser.
# http://sentiment.christopherpotts.net/tokenizing.html
# Note the addition of capture groups:
#   mouth:  mouth character of a conventional emoticon
#   rmouth: mouth character of a reversed emoticon.
emoticon_re = re.compile(r'''
    ^(?:
        [<>]?
        [:;=8]                     # eyes
        [\-o\*\']?                 # optional nose
        (?P<mouth>[\)\]\(\[dDpP/:\}\{@\|\\]) # mouth
    |
        (?P<rmouth>[\)\]\(\[dDpP/:\}\{@\|\\]) # mouth
        [\-o\*\']?                 # optional nose
        [:;=8]                     # eyes
        [<>]?
    )$
''', re.VERBOSE)

ALWAYS_UNHAPPY = frozenset(r'/\|S$')
HAPPY_MOUTHS = frozenset(r')]D}')
HAPPY_RMOUTHS = frozenset(r'([{')
UNHAPPY_MOUTHS = HAPPY_RMOUTHS | ALWAYS_UNHAPPY
UNHAPPY_RMOUTHS = HAPPY_MOUTHS | ALWAYS_UNHAPPY


class EmoticonType:
    UNHAPPY = 0
    NEUTRAL = 1
    HAPPY = 2


class Emoticons(BaseEstimator):
    '''
    Positive and negative emoticons.
    '''

    @staticmethod
    def assess_match(match):
        if match:
            groups = match.groupdict()
            if groups['mouth'] in HAPPY_MOUTHS:
                return EmoticonType.HAPPY
            elif groups['mouth'] in UNHAPPY_MOUTHS:
                return EmoticonType.UNHAPPY
            elif groups['rmouth'] in HAPPY_RMOUTHS:
                return EmoticonType.HAPPY
            elif groups['rmouth'] in UNHAPPY_RMOUTHS:
                return EmoticonType.UNHAPPY
        return EmoticonType.NEUTRAL

    def fit(self, docs, y=None):
        return self

    @reiterable
    def transform(self, docs):
        for doc in docs:
            matches = np.fromiter((self.assess_match(emoticon_re.match(word)) for word in doc), dtype='int32')
            yield np.hstack([matches == EmoticonType.UNHAPPY, matches == EmoticonType.HAPPY])

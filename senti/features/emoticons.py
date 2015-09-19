
import re

import numpy as np
from sklearn.base import BaseEstimator

from senti.base import ReiterableMixin

__all__ = ['Emoticons']

# Adapted from Christopher Potts' emoticon recognising tokeniser.
# http://sentiment.christopherpotts.net/tokenizing.html
# Note the addition of capture groups:
#   mouth:  mouth character of a conventional emoticon
#   rmouth: mouth character of a reversed emoticon.
emoticon_string = r'''
    ^(?:
        [<>]?
        [:;=8]                     # eyes
        [\-o\*\']?                 # optional nose
        (?P<mouth>[\)\]\(\[dDpP/\:\}\{@\|\\]) # mouth
    |
        (?P<rmouth>[\)\]\(\[dDpP/\:\}\{@\|\\]) # mouth
        [\-o\*\']?                 # optional nose
        [:;=8]                     # eyes
        [<>]?
    )$
'''

re_options = re.VERBOSE | re.I | re.UNICODE
emoticon_re = re.compile(emoticon_string, re_options)

ALWAYS_UNHAPPY = frozenset(r'/\|S$')
HAPPY_MOUTHS = frozenset(r')]D}')
HAPPY_RMOUTHS = frozenset(r'([{')
UNHAPPY_MOUTHS = HAPPY_RMOUTHS | ALWAYS_UNHAPPY
UNHAPPY_RMOUTHS = HAPPY_MOUTHS | ALWAYS_UNHAPPY


class Symbol:
    UNHAPPY = 0
    NEUTRAL = 1
    HAPPY = 2


class Emoticons(BaseEstimator, ReiterableMixin):
    '''
    Positive and negative emoticons.
    '''

    @staticmethod
    def assess_match(match):
        if match:
            groups = match.groupdict()
            if groups['mouth'] in HAPPY_MOUTHS:
                return Symbol.HAPPY
            elif groups['mouth'] in UNHAPPY_MOUTHS:
                return Symbol.UNHAPPY
            elif groups['rmouth'] in HAPPY_RMOUTHS:
                return Symbol.HAPPY
            elif groups['rmouth'] in UNHAPPY_RMOUTHS:
                return Symbol.UNHAPPY
        return Symbol.NEUTRAL

    def fit(self, docs, y=None):
        return self

    def _transform(self, docs):
        for doc in docs:
            matches = np.fromiter((self.assess_match(emoticon_re.match(word)) for word in doc), dtype='int32')
            yield np.hstack([matches == Symbol.UNHAPPY, matches == Symbol.NEUTRAL, matches == Symbol.HAPPY])

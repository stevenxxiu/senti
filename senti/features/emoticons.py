
import re
import numpy as np
from sklearn.base import BaseEstimator
from collections import Counter

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

HAPPY_SYMBOL = 'EMOT+'
UNHAPPY_SYMBOL = 'EMOT-'
NEUTRAL_SYMBOL = 'NA'


class Emoticons(BaseEstimator):
    '''
    Proportion of emoticons.
    '''

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def assess_match(match):
        if match:
            groups = match.groupdict()
            if groups['mouth'] in HAPPY_MOUTHS:
                return HAPPY_SYMBOL
            elif groups['mouth'] in UNHAPPY_MOUTHS:
                return UNHAPPY_SYMBOL
            elif groups['rmouth'] in HAPPY_RMOUTHS:
                return HAPPY_SYMBOL
            elif groups['rmouth'] in UNHAPPY_RMOUTHS:
                return UNHAPPY_SYMBOL
        return NEUTRAL_SYMBOL

    def fit(self, docs, y):
        return self

    def transform(self, docs):
        vecs = []
        for doc in docs:
            counts = Counter()
            symbol = None
            for token in self.tokenizer(doc):
                symbol = self.assess_match(emoticon_re.match(token))
                counts[symbol] += 1
            vecs.append(np.array([
                counts[HAPPY_SYMBOL], counts[UNHAPPY_SYMBOL], counts[NEUTRAL_SYMBOL],
                int(symbol == HAPPY_SYMBOL), int(symbol == UNHAPPY_SYMBOL), int(symbol == NEUTRAL_SYMBOL)
            ]))
        return np.vstack(vecs)

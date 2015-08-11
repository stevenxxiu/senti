
import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['AllCaps']


class AllCaps(BaseEstimator):
    '''
    Proportion of tokens with fully capitalised letters.
    '''

    def __init__(self, preprocessor, tokenizer):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    def fit(self, docs, y):
        return self

    def transform(self, docs):
        rows = []
        for doc in docs:
            tokens = self.tokenizer(self.preprocessor(doc))
            rows.append(sum(1 for token in tokens if token.isupper())/len(tokens))
        return np.vstack(rows)

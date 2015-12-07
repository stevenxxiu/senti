
from sklearn.base import BaseEstimator

from senti.utils import reiterable

__all__ = ['AsCorporas', 'MapCorporas', 'MergeSliceCorporas']


class AsCorporas(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def fit(self, docs, y=None):
        self.estimator.fit([docs])
        return self

    def transform(self, docs):
        return self.estimator.transform([docs])[0]


class MapCorporas(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def fit(self, corporas, y=None):
        for corpora in corporas:
            self.estimator.fit(corpora, y)
        return self

    def transform(self, corporas):
        return list(map(self.estimator.transform, corporas))


class MergeSliceCorporas(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self._corporas = []
        self._corporas_start = []
        self._corporas_end = []

    @reiterable
    def _chain_docs(self, corporas):
        self._corporas = corporas
        pos = 0
        for corpora in corporas:
            self._corporas_start.append(pos)
            i = -1
            for i, doc in enumerate(corpora):
                yield doc
            pos += i + 1
            self._corporas_end.append(pos)

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    def fit(self, corporas, y=None):
        self.estimator.fit(self._chain_docs(corporas), y)
        return self

    def transform(self, corporas):
        transformed = self.estimator.transform(None)
        res = []
        for corpora in corporas:
            try:
                i = self._corporas.index(corpora)
            except IndexError:
                raise ValueError('docs were not fitted')
            res.append(transformed[self._corporas_start[i]:self._corporas_end[i]])
        return res

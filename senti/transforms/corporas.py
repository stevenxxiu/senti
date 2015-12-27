
from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.sklearn_ import skip_empty_fit

__all__ = ['AsCorporas', 'MapCorporas', 'MergeSliceCorporas']


class AsCorporas(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    @skip_empty_fit
    def fit(self, docs, y=None):
        self.estimator.fit([docs])
        return self

    def transform(self, docs):
        return next(iter(self.estimator.transform([docs])))


class MapCorporas(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        return self.estimator.get_params(deep)

    @skip_empty_fit
    def fit(self, corporas, y=None):
        for corpora in corporas:
            self.estimator.fit(corpora, y)
        return self

    @reiterable
    def transform(self, corporas):
        yield from map(self.estimator.transform, corporas)


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

    @skip_empty_fit
    def fit(self, corporas, y=None):
        self.estimator.fit(self._chain_docs(corporas), y)
        return self

    @reiterable
    def transform(self, corporas):
        transformed = self.estimator.transform(None)
        if not hasattr(transformed, '__index__'):
            transformed = list(transformed)
        for corpora in corporas:
            try:
                i = self._corporas.index(corpora)
            except IndexError:
                raise ValueError('docs were not fitted')
            yield transformed[self._corporas_start[i]:self._corporas_end[i]]

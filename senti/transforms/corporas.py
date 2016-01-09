
from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.sklearn_ import skip_empty_fit

__all__ = ['ToCorporas', 'MapCorporas', 'MergeSliceCorporas', 'MultiInputOutput']


class ToCorporas(BaseEstimator):
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


class MultiInputOutput(BaseEstimator):
    def __init__(self, estimator, inputs, input_names, output_name):
        self.estimator = estimator
        self.inputs = inputs
        self.input_names = input_names
        self.output_name = output_name

    @skip_empty_fit
    def fit(self, data, *args, **kwargs):
        for input_name in self.input_names:
            kwargs[input_name] = {key: self.inputs[key].transform(value) for key, value in kwargs[input_name].items()}
        self.estimator.fit({key: self.inputs[key].transform(value) for key, value in data.items()}, *args, **kwargs)
        return self

    def transform(self, data):
        return self.estimator.transform(
            {key: self.inputs[key].transform(value) for key, value in data.items()}
        )[self.output_name]

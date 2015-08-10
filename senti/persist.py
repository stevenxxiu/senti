
from abc import ABCMeta, abstractmethod
from contextlib import suppress


class BaseEstimatorWrapper(metaclass=ABCMeta):
    '''
    Base class for estimator wappers that delegate get_params and attributes.
    '''

    @abstractmethod
    def __init__(self, estimator):
        self._setattr('_estimator', estimator)

    def _setattr(self, attr, value):
        super().__setattr__(attr, value)

    def __getattr__(self, attr):
        return getattr(self._estimator, attr)

    def __setattr__(self, attr, value):
        return setattr(self._estimator, attr, value)


class CachedFitTransform(BaseEstimatorWrapper):
    '''
    Optional hashing of the data is used for fit & transform for speed. This means that the same data may be cached
    multiple times on disk, given different hashes. But since the process of generating the data would likely be the
    same each time and this is what we hash, this is unlikely.

    Hash collisions are rare since joblib uses md5, so can be ignored.
    '''

    def __init__(self, estimator, memory, ignored_params=()):
        super().__init__(estimator)
        self._setattr('_fit_hash', None)
        self._setattr('_transform_hash', None)
        self._setattr('_cached_fit', memory.cache(self._cached_fit, ignore=['self', 'X_hash']))
        self._setattr('_cached_fit_hash', memory.cache(self._cached_fit, ignore=['self', 'X']))
        self._setattr('_cached_transform', memory.cache(self._cached_transform, ignore=['self', 'X_hash']))
        self._setattr('_cached_transform_hash', memory.cache(self._cached_transform, ignore=['self', 'X']))
        self._setattr('_memory', memory)
        self._setattr('_ignored_params', ignored_params)

    def _cached_fit(self, key_params, *args, X_hash=None, **kwargs):
        self._estimator.fit(*args, **kwargs)
        return self._estimator.__dict__

    def fit(self, X, *args, X_hash=None, **kwargs):
        ignored = {}
        key_params = self._estimator.get_params(deep=True)
        for param in self._ignored_params:
            with suppress(KeyError):
                ignored[param] = key_params.pop(param)
        fit_func = self._cached_fit if X_hash is None else self._cached_fit_hash
        self._estimator.__dict__, self._fit_hash, _ = \
            fit_func._cached_call(key_params, X, *args, X_hash=X_hash, **kwargs)
        if ignored:
            self._estimator.set_params(**ignored)
        return self

    def _cached_transform(self, key_params, X, X_hash=None):
        return self._estimator.transform(X)

    def transform(self, X, X_hash=None):
        transform_func = self._cached_transform if X_hash is None else self._cached_transform_hash
        res, self._transform_hash, _ = transform_func._cached_call(self._fit_hash, X, X_hash=X_hash)
        return res

    def fit_transform(self, X, *args, X_hash=None, **kwargs):
        # ignore the default fit_transform as using the cache is usually more efficient
        return self.fit(X, *args, X_hash=X_hash, **kwargs).transform(X, X_hash=X_hash)


from contextlib import suppress

from wrapt import ObjectProxy


class CachedFitTransform(ObjectProxy):
    '''
    Optional hashing of the data is used for fit & transform for speed. This means that the same data may be cached
    multiple times on disk, given different hashes. But since the process of generating the data would likely be the
    same each time and this is what we hash, this is unlikely.

    Hash collisions are rare since joblib uses md5, so can be ignored.
    '''

    def __init__(self, estimator, memory, ignored_params=()):
        super().__init__(estimator)
        self._self_fit_hash = None
        self._self_transform_hash = None
        self._self_cached_fit = memory.cache(self._cached_fit, ignore=['self', 'X_hash'])
        self._self_cached_fit_hash = memory.cache(self._cached_fit, ignore=['self', 'X'])
        self._self_cached_transform = memory.cache(self._cached_transform, ignore=['self', 'X_hash'])
        self._self_cached_transform_hash = memory.cache(self._cached_transform, ignore=['self', 'X'])
        self._self_memory = memory
        self._self_ignored_params = ignored_params

    @staticmethod
    def _cached_call(func, *args, **kwargs):
        # noinspection PyProtectedMember
        return func._cached_call(args, kwargs)

    def _cached_fit(self, key_params, X, X_hash, *args, **kwargs):
        self.__wrapped__.fit(X, *args, **kwargs)
        return self.__wrapped__.__dict__

    def fit(self, X, *args, **kwargs):
        ignored = {}
        key_params = self.__wrapped__.get_params(deep=True)
        for param in self._self_ignored_params:
            with suppress(KeyError):
                ignored[param] = key_params.pop(param)
        X_hash = getattr(X, 'joblib_hash', None) or getattr(X, '_self_joblib_hash', None)
        fit_func = self._self_cached_fit_hash if X_hash else self._self_cached_fit
        self.__wrapped__.__dict__, self._self_fit_hash, _ = \
            self._cached_call(fit_func, key_params, X, X_hash, *args, **kwargs)
        if ignored:
            self.__wrapped__.set_params(**ignored)
        return self

    def _cached_transform(self, fit_hash, X_hash, X):
        return self.__wrapped__.transform(X)

    def transform(self, X):
        X_hash = getattr(X, 'joblib_hash', None) or getattr(X, '_self_joblib_hash', None)
        transform_func = self._self_cached_transform_hash if X_hash else self._self_cached_transform
        res, res_hash, _ = self._cached_call(transform_func, self._self_fit_hash, X_hash, X)
        res = ObjectProxy(res)
        res._self_joblib_hash = res_hash
        return res

    def fit_transform(self, X, *args, **kwargs):
        # ignore the default fit_transform as using the cache is usually more efficient
        return self.fit(X, *args, **kwargs).transform(X)

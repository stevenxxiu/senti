
import itertools
import os
import pickle
from contextlib import suppress

from senti.utils.sklearn_ import EmptyFitMixin
from senti.utils.utils import PicklableProxy, split_every

__all__ = ['CachedFitTransform', 'CachedIterable']


# noinspection PyProtectedMember
class CachedFitTransform(PicklableProxy):
    '''
    Optional hashing of the data is used for fit & transform for speed. This means that the same data may be cached
    multiple times on disk, given different hashes. But since the process of generating the data would likely be the
    same each time and this is what we hash, this is unlikely.

    Hash collisions are rare since joblib uses md5, so can be ignored.
    '''

    def __init__(self, estimator, memory, ignored_params=()):
        super().__init__(estimator, memory, ignored_params)
        self._self_fit_hash = None
        self._self_transform_hash = None
        self._self_cached_fit = memory.cache(self._cached_fit, ignore=['self', 'X_hash'])
        self._self_cached_fit_hash = memory.cache(self._cached_fit, ignore=['self', 'X'])
        self._self_cached_transform = memory.cache(self._cached_transform, ignore=['self', 'X_hash'])
        self._self_cached_transform_hash = memory.cache(self._cached_transform, ignore=['self', 'X'])
        self._self_memory = memory
        self._self_ignored_params = ignored_params

    def _cached_fit(self, cls, key_params, X, X_hash, *args, **kwargs):
        self.__wrapped__.fit(X, *args, **kwargs)
        return self.__wrapped__.__dict__

    def fit(self, X, *args, **kwargs):
        if not isinstance(self.__wrapped__, EmptyFitMixin):
            ignored = {}
            key_params = self.__wrapped__.get_params(deep=True)
            for param in self._self_ignored_params:
                with suppress(KeyError):
                    ignored[param] = key_params.pop(param)
            X_hash = getattr(X, 'joblib_hash', None) or getattr(X, '_self_joblib_hash', None)
            fit_func = self._self_cached_fit_hash if X_hash else self._self_cached_fit
            # don't use the unwrapped object as users may also use ObjectProxy
            self.__wrapped__.__dict__, self._self_fit_hash, _ = \
                fit_func._cached_call([type(self.__wrapped__), key_params, X, X_hash, *args], kwargs)
        return self

    def _cached_transform(self, cls, fit_hash, X_hash, X):
        res = self.__wrapped__.transform(X)
        return CachedIterable(res) if hasattr(res, '__iter__') and not hasattr(res, '__len__') else res

    def transform(self, X):
        X_hash = getattr(X, 'joblib_hash', None) or getattr(X, '_self_joblib_hash', None)
        transform_func = self._self_cached_transform_hash if X_hash else self._self_cached_transform
        res, res_hash, _ = transform_func._cached_call([type(self.__wrapped__), self._self_fit_hash, X_hash, X], {})
        if not isinstance(res, PicklableProxy):
            res = PicklableProxy(res)
        res._self_joblib_hash = res_hash
        return res

    def fit_transform(self, X, *args, **kwargs):
        # ignore the default fit_transform as using the cache is usually more efficient
        return self.fit(X, *args, **kwargs).transform(X)


class CachedIterable(PicklableProxy):
    def __init__(self, wrapped, chunk_size=100):
        super().__init__(wrapped)
        self._self_chunk_size = chunk_size
        self._self_name = None
        self._self_path = 'cache/joblib_gen'
        if not os.path.exists(self._self_path):
            os.makedirs(self._self_path)

    def __iter__(self):
        if self._self_name is None:
            yield from self.__wrapped__
        else:
            with open(os.path.join(self._self_path, self._self_name), 'rb') as sr:
                while True:
                    try:
                        yield from pickle.load(sr)
                    except EOFError:
                        break

    def __reduce__(self):
        if self._self_name is None:
            name = None
            for i in itertools.count(1):
                name = 'output.pkl_{:02}.gen'.format(i)
                if not os.path.exists(os.path.join(self._self_path, name)):
                    break
            with open(os.path.join(self._self_path, name), 'wb') as sr:
                for chunk in split_every(self, self._self_chunk_size):
                    pickle.dump(chunk, sr)
            self._self_name = name
        return super().__reduce__()

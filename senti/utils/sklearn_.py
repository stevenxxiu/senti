
from contextlib import contextmanager

__all__ = ['EmptyFitMixin', 'has_empty_fit', 'skip_empty_fit']


class EmptyFitMixin:
    def fit(self, X, y=None):
        return self


def has_empty_fit(estimator):
    if isinstance(estimator, EmptyFitMixin):
        return True
    if hasattr(estimator, 'steps') and all(has_empty_fit(e) for _, e in estimator.steps):
        return True
    if hasattr(estimator, 'estimator'):
        return has_empty_fit(estimator.estimator)
    return False


def skip_empty_fit(func):
    def decorated(self, *args, **kwargs):
        if has_empty_fit(self):
            return self
        return func(self, *args, **kwargs)

    return decorated


__all__ = ['EmptyFitMixin', 'is_fit_empty']


class EmptyFitMixin:
    def fit(self, X, y=None):
        return self


def is_fit_empty(estimator):
    if isinstance(estimator, EmptyFitMixin):
        return True
    if hasattr(estimator, 'steps') and all(isinstance(e, EmptyFitMixin) for _, e in estimator.steps):
        return True
    return False


__all__ = ['EmptyFitMixin']


class EmptyFitMixin:
    def fit(self, X, y=None):
        return self


import itertools

import numpy as np
from scipy import sparse

__all__ = ['sparse_sum', 'vstack']


def sparse_sum(X, axis):
    if axis == 0:
        n = X.shape[0]
        return sparse.csr_matrix((np.ones(n), np.arange(n), (0, n)))*X
    elif axis == 1:
        n = X.shape[1]
        return X*sparse.csr_matrix((np.ones(n), np.arange(n), (0, n)))


def vstack(Xs):
    Xs = iter(Xs)
    X = next(Xs)
    Xs = itertools.chain([X], Xs)
    return sparse.vstack(Xs) if sparse.issparse(X) else np.vstack(Xs)


import itertools
import json
import sys

import numpy as np
from scipy import sparse
from wrapt import ObjectProxy

__all__ = [
    'Tee', 'PicklableSr', 'FieldExtractor', 'HeadSr', 'PicklableProxy', 'reiterable', 'compose', 'sparse_sum', 'vstack'
]


class Tee:
    def __init__(self, *args, **kwargs):
        self.file = open(*args, **kwargs)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


class PicklableSr:
    def __init__(self, sr):
        self.sr = sr
        self.name = sr.name
        self.encoding = sr.encoding

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['sr']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sr = open(self.name, encoding=self.encoding)


class FieldExtractor(PicklableSr):
    def __init__(self, sr, field):
        super().__init__(sr)
        self.field = field

    def __iter__(self):
        self.sr.seek(0)
        for line in self.sr:
            yield json.loads(line)[self.field]


class HeadSr(PicklableSr):
    def __init__(self, sr, n):
        super().__init__(sr)
        self.n = n

    def __iter__(self):
        self.sr.seek(0)
        for i, line in zip(range(self.n), self.sr):
            yield line


class PicklableProxy(ObjectProxy):
    def __init__(self, wrapped, *args):
        super().__init__(wrapped)
        self._self_attrs = {'_self_attrs'}
        self._self_args = args

    def __setattr__(self, name, value):
        if name.startswith('_self_') and name != '_self_attrs':
            self._self_attrs.add(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name.startswith('_self_') and name != '_self_attrs':
            self._self_attrs.remove(name)
        super().__delattr__(name)

    def __reduce__(self):
        return type(self), (self.__wrapped__,) + self._self_args, \
            tuple((attr, getattr(self, attr)) for attr in sorted(self._self_attrs) if attr != '_self_attrs')

    def __setstate__(self, state):
        for attr, value in state:
            setattr(self, attr, value)


class Reiterable:
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def __hash__(self):
        return hash((self.func, self.args))

    def __eq__(self, other):
        return self.func == other.func and self.args == other.args

    def __iter__(self):
        yield from self.func(*self.args)


def reiterable(method):
    def decorated(*args):
        return Reiterable(method, *args)

    return decorated


class Compose:
    def __init__(self, funcs):
        self.funcs = funcs

    def __eq__(self, other):
        return self.funcs == other.funcs

    def __call__(self, *args, **kwargs):
        res = self.funcs[-1](*args, **kwargs)
        for func in self.funcs[-2::-1]:
            res = func(res)
        return res

compose = Compose


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

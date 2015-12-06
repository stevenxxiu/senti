
import logging
from contextlib import contextmanager

from wrapt import ObjectProxy

__all__ = ['PicklableProxy', 'reiterable', 'compose', 'temp_log_level']


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
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.func(*self.args, **self.kwargs)

    def __eq__(self, other):
        try:
            return self.func == other.func and self.args == other.args and self.kwargs == other.kwargs
        except AttributeError:
            return False


def reiterable(method):
    def decorated(*args, **kwargs):
        return Reiterable(method, *args, **kwargs)

    return decorated


class Compose:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __eq__(self, other):
        try:
            return self.funcs == other.funcs
        except AttributeError:
            return False

    def __call__(self, *args, **kwargs):
        res = self.funcs[-1](*args, **kwargs)
        for func in self.funcs[-2::-1]:
            res = func(res)
        return res

compose = Compose


@contextmanager
def temp_log_level(loggers_levels):
    prev_levels = {}
    for logger, level in loggers_levels.items():
        prev_levels[logger] = logging.getLogger(logger).getEffectiveLevel()
        logging.getLogger(logger).setLevel(level)
    yield
    for logger, level in prev_levels.items():
        logging.getLogger(logger).setLevel(level)


import itertools
import logging
import os
import re
import time
from contextlib import contextmanager

from wrapt import ObjectProxy

__all__ = ['PicklableProxy', 'reiterable', 'compose', 'snake_case', 'split_every', 'temp_chdir', 'temp_log_level', 'log_time']


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


def snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def split_every(iter_, n):
    iter_ = iter(iter_)
    while True:
        res = list(itertools.islice(iter_, n))
        if not res:
            break
        yield res


@contextmanager
def temp_chdir(path):
    prev_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(prev_dir)


@contextmanager
def temp_log_level(loggers_levels):
    prev_levels = {}
    for logger, level in loggers_levels.items():
        prev_levels[logger] = logging.getLogger(logger).getEffectiveLevel()
        logging.getLogger(logger).setLevel(level)
    yield
    for logger, level in prev_levels.items():
        logging.getLogger(logger).setLevel(level)


@contextmanager
def log_time(fmt_start, fmt_end):
    logging.info(fmt_start)
    start = time.time()
    yield
    end = time.time()
    logging.info(fmt_end.format(end - start))

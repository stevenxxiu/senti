
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock

import joblib
from joblib import Memory
from sklearn.base import BaseEstimator

from senti.utils import *

# Mock is not used in the usual way since it runs into pickling problems.
_mock = MagicMock()


@contextmanager
def set_mock(cb):
    with cb() as mock:
        global _mock
        _mock = mock
        yield mock


class EsimatorA(BaseEstimator):
    def __init__(self, param):
        self.param = param
        self.X = 0

    def fit(self, X):
        self.X = X
        _mock()

    def transform(self, X):
        _mock()
        return self.param


class EsimatorB(EsimatorA):
    pass


class TestCachedFitTransform(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.memory = Memory(self.tempdir.name, verbose=0)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_fit(self):
        obj = CachedFitTransform(EsimatorA(1), self.memory)
        obj.fit(1)
        self.assertEqual(obj.X, 1)
        obj.fit(2)
        self.assertEqual(obj.X, 2)
        obj = CachedFitTransform(EsimatorA(1), self.memory)
        with set_mock(MagicMock) as mock:
            obj.fit(1)
            self.assertEqual(obj.X, 1)
            self.assertEqual(mock.call_count, 0)
        with set_mock(MagicMock) as mock:
            obj.fit(2)
            self.assertEqual(obj.X, 2)
            self.assertEqual(mock.call_count, 0)

    def test_fit_diff_param(self):
        obj = CachedFitTransform(EsimatorA(1), self.memory)
        obj.fit(1)
        obj = CachedFitTransform(EsimatorA(2), self.memory)
        with set_mock(MagicMock) as mock:
            obj.fit(1)
            self.assertEqual(obj.X, 1)
            self.assertEqual(mock.call_count, 1)

    def test_fit_diff_cls(self):
        obj = CachedFitTransform(EsimatorA(1), self.memory)
        obj.fit(1)
        obj = CachedFitTransform(EsimatorB(2), self.memory)
        with set_mock(MagicMock) as mock:
            obj.fit(1)
            self.assertEqual(obj.X, 1)
            self.assertEqual(mock.call_count, 1)

    def test_fit_ignore_param(self):
        obj = CachedFitTransform(EsimatorA(1), self.memory, ignored_params=['param'])
        obj.fit(1)
        obj = CachedFitTransform(EsimatorA(2), self.memory, ignored_params=['param'])
        with set_mock(MagicMock) as mock:
            obj.fit(1)
            self.assertEqual(obj.X, 1)
            self.assertEqual(mock.call_count, 0)

    def test_transform(self):
        obj = CachedFitTransform(EsimatorA(1), self.memory, ignored_params=['param'])
        self.assertEqual(obj.transform(1), 1)
        with set_mock(MagicMock) as mock:
            self.assertEqual(obj.transform(1), 1)
            self.assertEqual(mock.call_count, 0)

    def test_transform_hash(self):
        obj = CachedFitTransform(EsimatorA(1), self.memory, ignored_params=['param'])
        X = MagicMock()
        X.joblib_hash = 1
        self.assertEqual(obj.transform(X), 1)
        with set_mock(MagicMock) as mock:
            X = MagicMock()
            X.joblib_hash = 1
            self.assertEqual(obj.transform(X), 1)
            self.assertEqual(mock.call_count, 0)


# noinspection PyUnresolvedReferences
class TestCachedIterable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        os.chdir(cls.dir.name)

    @classmethod
    def tearDownClass(cls):
        os.chdir('..')
        cls.dir.cleanup()

    @reiterable
    def iterator(self):
        yield from range(11)

    def test_pickle(self):
        joblib.dump(CachedIterable(self.iterator(), 2), 'output')
        self.assertListEqual(list(joblib.load('output')), list(range(11)))

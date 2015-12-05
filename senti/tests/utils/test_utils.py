
import os
import joblib
import pickle
import tempfile
import unittest

from senti.utils import *


class TestPicklableProxy(unittest.TestCase):
    def test_pickle(self):
        obj = PicklableProxy(1)
        obj._self_value = 2
        obj = pickle.loads(pickle.dumps(obj))
        self.assertEqual(obj, 1)
        self.assertEqual(obj._self_value, 2)


# noinspection PyUnresolvedReferences
class TestReiterable(unittest.TestCase):
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
        joblib.dump(reiterable(self.iterator)(chunk_size=2), 'output')
        self.assertListEqual(list(joblib.load('output')), list(range(11)))

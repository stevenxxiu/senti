
import pickle
import unittest

from senti.utils import *


class TestPicklableProxy(unittest.TestCase):
    def test_pickle(self):
        obj = PicklableProxy(1)
        obj._self_value = 2
        obj = pickle.loads(pickle.dumps(obj))
        self.assertEqual(obj, 1)
        self.assertEqual(obj._self_value, 2)

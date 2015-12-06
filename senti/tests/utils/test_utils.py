
import pickle
import unittest

from senti.utils.utils import *


class TestPicklableProxy(unittest.TestCase):
    def test_set_attr(self):
        obj = PicklableProxy(1)
        obj._self_attr = 2
        obj_loaded = pickle.loads(pickle.dumps(obj))
        self.assertEqual(obj_loaded, obj)
        self.assertEqual(obj_loaded, 1)
        self.assertEqual(obj_loaded._self_attr, 2)

    def test_del_attr(self):
        obj = PicklableProxy(1)
        obj._self_attr = 2
        del obj._self_attr
        obj_loaded = pickle.loads(pickle.dumps(obj))
        self.assertEqual(obj_loaded, obj)
        self.assertEqual(obj_loaded, 1)
        self.assertFalse(hasattr(obj_loaded, '_self_attr'))


class TestReiterable(unittest.TestCase):
    @reiterable
    def iterator(self, n):
        yield from range(n)

    def test_iter(self):
        iter_ = self.iterator(3)
        self.assertListEqual(list(iter_), list(range(3)))
        self.assertListEqual(list(iter_), list(range(3)))

    def test_eq(self):
        self.assertEqual(self.iterator(3), self.iterator(3))
        self.assertNotEqual(self.iterator(3), self.iterator(4))
        self.assertNotEqual(self.iterator(3), None)


class TestCompose(unittest.TestCase):
    def test_compose(self):
        self.assertEqual(compose(lambda x: x + 1, lambda x: x + 1, lambda x, y: x + y)(1, 2), 5)

    def test_eq(self):
        self.assertEqual(compose(1, 2), compose(1, 2))
        self.assertNotEqual(compose(1, 2), compose(1))
        self.assertNotEqual(compose(1, 2), 1)

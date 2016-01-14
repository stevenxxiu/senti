
import unittest

import numpy as np

from senti.utils.numpy_ import *


class TestNumpy(unittest.TestCase):
    def test_clip_pad(self):
        self.assertTrue(np.array_equal(clippad(np.array([1, 2, 3]), 3), np.array([1, 2, 3])))

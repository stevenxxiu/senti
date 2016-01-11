
import itertools
import unittest

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import *

from senti.utils.lasagne_ import *


class TestKMaxPool1DLayer(unittest.TestCase):
    def test_get_output_shape_for(self):
        self.assertEqual(
            KMaxPool1DLayer(InputLayer((100, 100)), 3).get_output_shape_for((10, 20, 30)),
            (10, 20, 3)
        )

    def test_get_output_for(self):
        X = T.itensor3()
        X1 = np.empty((2, 2, 10), dtype='int32')
        for i, is_ in enumerate(itertools.product(*(range(n) for n in X1.shape[:-1]))):
            X1[is_] = np.arange(i, 10 + i)
        X2 = np.empty((2, 2, 3), dtype='int32')
        for i, is_ in enumerate(itertools.product(*(range(n) for n in X2.shape[:-1]))):
            X2[is_] = np.arange(7 + i, 10 + i)
        self.assertTrue(np.array_equal(
            theano.function([X], KMaxPool1DLayer(InputLayer((100, 100)), 3).get_output_for(X))(X1), X2
        ))

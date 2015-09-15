
import lasagne
import numpy.random as random

__all__ = ['get_rng', 'set_rng']

_rng = random


def get_rng():
    # prevents the rng being accessed as an import
    return _rng


def set_rng(new_rng):
    global _rng
    _rng = new_rng
    lasagne.random.set_rng(_rng)

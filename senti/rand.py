
import lasagne
import numpy as np

__all__ = ['get_rng', 'seed_rng']

_rng = np.random


def get_rng():
    # prevents the rng being accessed as an import
    return _rng


def seed_rng(seed):
    global _rng
    _rng = np.random.RandomState(seed)
    lasagne.random.set_rng(_rng)


import sys

import numpy as np

__all__ = ['Tee', 'Compose', 'indexes_of']


class Tee(object):
    def __init__(self, *args, **kwargs):
        self.file = open(*args, **kwargs)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


class Compose:
    '''
    Picklable compose implementation.
    '''

    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, *args, **kwargs):
        res = self.funcs[-1](*args, **kwargs)
        for func in self.funcs[1::-1]:
            res = func(res)
        return res


def indexes_of(x, y):
    '''
    Find the indexes of y in x.
    '''
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    return np.take(index, sorted_index, mode='clip')

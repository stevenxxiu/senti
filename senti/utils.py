
import numpy as np

__all__ = ['indexes_of']


def indexes_of(x, y):
    '''
    Find the indexes of y in x.
    '''
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    return np.take(index, sorted_index, mode='clip')

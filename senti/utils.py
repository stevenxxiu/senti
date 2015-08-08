
import os
from json import JSONEncoder

import numpy as np
from scipy.sparse import coo_matrix

__all__ = ['third_dir', 'obj_fullname', 'indexes_of', 'SciPyJSONEncoder', 'decode_scipy_object']

third_dir = os.path.join(os.path.dirname(__file__), '../third')


def obj_fullname(o):
    t = type(o)
    return t.__module__ + '.' + t.__name__


def indexes_of(x, y):
    '''
    Find the indexes of y in x.
    '''
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    return np.take(index, sorted_index, mode='clip')


class SciPyJSONEncoder(JSONEncoder):
    '''
    Encodes numpy & scipy arrays. This is much less verbose than jsonpickle.
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'py/object': obj_fullname(obj), 'data': obj.tolist()}
        elif isinstance(obj, coo_matrix):
            return {
                'py/object': obj_fullname(obj), 'data': obj.data, 'row': obj.row, 'col': obj.col, 'shape': obj.shape
            }
        return super().default(obj)


def decode_scipy_object(obj):
    obj_type = obj.get('py/object', None)
    if obj_type == 'numpy.ndarray':
        return np.array(obj['data'])
    elif obj_type == 'scipy.sparse.coo.coo_matrix':
        return coo_matrix((obj['data'], (obj['row'], obj['col'])), obj['shape'])
    else:
        return obj

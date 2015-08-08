
import numpy as np

from senti.stream import PersistableStream

__all__ = ['AllCaps']


class AllCaps(PersistableStream):
    '''
    Counts # of tokens with fully capitalised letters.
    '''

    def __init__(self, src_sr, reuse=False):
        super().__init__('allcaps({})'.format(src_sr.name), (src_sr,), reuse)

    def _iter(self):
        for obj in self.src_srs[0]:
            tokens = obj.pop('tokens')
            obj['vec'] = np.array([sum(1 for token in tokens if token.isupper())])
            yield obj

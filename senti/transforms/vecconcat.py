
from scipy import sparse

from senti.stream import PersistableStream

__all__ = ['VecConcatTransform']


class VecConcatTransform(PersistableStream):
    def __init__(self, *src_srs, reuse=False):
        super().__init__('concat({})'.format(','.join(src_sr.name for src_sr in src_srs)), src_srs, reuse)

    def _iter(self):
        for objs in zip(*self.src_srs):
            objs[0]['vec'] = sparse.hstack(tuple(obj['vec'] for obj in objs))
            yield objs[0]

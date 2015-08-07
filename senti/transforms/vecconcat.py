
from senti.stream import PersistableStream

__all__ = ['VecConcatTransform']


class VecConcatTransform(PersistableStream):
    def __init__(self, *src_srs, reuse=False):
        super().__init__('concat({})'.format(','.join(src_sr.name for src_sr in src_srs)), src_srs, reuse)

    def _iter(self):
        for objs in zip(*self.src_srs):
            vec = []
            for obj in objs:
                vec.extend(obj['vec'])
            obj['vec'] = vec
            yield obj

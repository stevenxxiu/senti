
from senti.stream import PersistableStream

__all__ = ['LowerTransform']


class LowerTransform(PersistableStream):
    def __init__(self, src_sr, reuse=False):
        super().__init__('lower({})'.format(src_sr.name), (src_sr,), reuse=reuse)

    def _iter(self):
        for obj in self.src_srs[0]:
            obj['tokens'] = list(map(str.lower, obj['tokens']))
            yield obj

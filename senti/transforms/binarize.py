
from senti.stream import PersistableStream

__all__ = ['BinarizeTransform']


class BinarizeTransform(PersistableStream):
    '''
    Binarize the vec.
    '''

    def __init__(self, src_sr, reuse=False):
        super().__init__('binarize({})'.format(src_sr.name), (src_sr,), reuse)

    def _iter(self):
        for obj in self.src_srs[0]:
            vec = obj['vec']
            yield {'id': obj['id'], 'vec': list(int(bool(x)) for x in vec)}

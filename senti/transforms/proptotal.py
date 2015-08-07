
from senti.stream import PersistableStream

__all__ = ['PropTotalTransform']


class PropTotalTransform(PersistableStream):
    '''
    Normalize the vec using l1 distance.
    '''

    def __init__(self, src_sr, reuse=False):
        super().__init__('prop_total({})'.format(src_sr.name), (src_sr,), reuse)

    def _iter(self):
        for obj in self.src_srs[0]:
            vec = obj['vec']
            l = sum(vec)
            yield {'id': obj['id'], 'vec': list(x/l for x in vec)}

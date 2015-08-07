
from senti.stream import PersistableStream

__all__ = ['PropTokensTransform']


class PropTokensTransform(PersistableStream):
    '''
    Divide by the # of tokens.
    '''

    def __init__(self, src_sr, reuse=False):
        super().__init__('prop_tokens({})'.format(src_sr.name), (src_sr,), reuse)

    def _iter(self):
        for obj in self.src_srs[0]:
            vec = obj['vec']
            l = obj['ntokens']
            yield {'id': obj['id'], 'vec': list(x/l for x in vec)}

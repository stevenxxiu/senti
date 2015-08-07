
import re

from senti.stream import PersistableStream

__all__ = ['NormalizeTransform']


class NormalizeTransform(PersistableStream):
    '''
    Normalizes & tokenizes.
    '''

    def __init__(self, src_sr, reuse=False):
        super().__init__('normalize({})'.format(src_sr.name), (src_sr,), reuse=reuse)

    def _iter(self):
        for obj in self.src_srs[0]:
            text = obj.pop('text')
            text = re.sub(r'\S{,4}://\S+', '_URL', text)
            text = re.sub(r'([\.\",()!?;:])', r' \1 ', text)
            tokens = text.strip().split()
            obj.update({'tokens': tokens, 'ntokens': len(tokens)})
            yield obj

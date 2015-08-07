
import re

from senti.stream import PersistableStream


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'([\.\",()!?;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class NormTextStream(PersistableStream):
    def __init__(self, src_sr, reuse=False):
        super().__init__('{}.norm'.format(src_sr.name), (src_sr,), reuse=reuse)

    def _iter(self):
        for obj in self.src_srs[0]:
            obj['text'] = normalize_text(obj['text'])
            yield obj

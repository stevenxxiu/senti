
from senti.stream import PersistableStream

__all__ = ['PersistentTransform']


class PersistentTransform:
    def __init__(self, name, src_srs=(), reuse_options=None):
        self.name = name
        self.src_srs = src_srs
        self.reuse_options = reuse_options
        self.train_sr = None

    @staticmethod
    def persist(method):
        def decorated(self, src_sr, *args, reuse=True, **kwargs):
            srs = (src_sr,) + self.src_srs
            name = '{}_{}({})'.format(
                self.name, {'fit': 'model', 'transform': 'features'}[method.__name__], ','.join(sr.name for sr in srs)
            )
            return PersistableStream(name, method(self, src_sr, *args, **kwargs), srs, reuse, self.reuse_options)
        return decorated

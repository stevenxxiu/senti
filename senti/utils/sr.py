
import itertools
import json
import sys
from contextlib import contextmanager

__all__ = ['RepeatSr', 'Tee', 'JSONDecoder', 'FieldExtractor', 'BalancedSlice']


@contextmanager
def reset_sr(*srs):
    # seek after yield to help with joblib caching of input arguments
    for sr in srs:
        sr.seek(0)
    yield
    for sr in srs:
        sr.seek(0)


class RepeatSr(itertools.repeat):
    def seek(self, pos, mode=0):
        pass


class Tee:
    def __init__(self, *args, **kwargs):
        self.file = open(*args, **kwargs)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


class JSONDecoder:
    def __init__(self, sr):
        self.sr = sr

    def __iter__(self):
        with reset_sr(self.sr):
            for line in self.sr:
                yield json.loads(line)


class FieldExtractor:
    def __init__(self, sr, field):
        self.sr = sr
        self.field = field

    def __iter__(self):
        for obj in self.sr:
            yield obj[self.field]


class BalancedSlice:
    def __init__(self, srs, n=None):
        self.srs = srs
        self.n = n

    def __iter__(self):
        with reset_sr(*self.srs):
            m = None if self.n is None else round(self.n / len(self.srs))
            m_last = None if self.n is None else self.n - m * (len(self.srs) - 1)
            for i, sr in enumerate(self.srs):
                if m is None:
                    yield from sr
                else:
                    yield from itertools.islice(sr, m if i < len(self.srs) - 1 else m_last)

    def __getitem__(self, item):
        if item.start is not None or item.stop < 0 or item.step is not None:
            raise ValueError('only slicing from the start with step 1 is supported')
        return BalancedSlice(self.srs, item.stop if self.n is None else min(item.stop, self.n))

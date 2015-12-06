
import json
import sys

__all__ = ['Tee', 'PicklableSr', 'FieldExtractor', 'BalancedSlice']


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


class PicklableSr:
    def __init__(self, sr):
        self.sr = sr
        self.name = sr.name
        self.encoding = sr.encoding

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['sr']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sr = open(self.name, encoding=self.encoding)


class FieldExtractor(PicklableSr):
    def __init__(self, sr, field):
        super().__init__(sr)
        self.field = field

    def __iter__(self):
        self.sr.seek(0)
        for line in self.sr:
            yield json.loads(line)[self.field]


class BalancedSlice:
    def __init__(self, srs, n=None):
        self.srs = srs
        self.n = n

    def __iter__(self):
        for sr in self.srs:
            sr.seek(0)
        m = None if self.n is None else round(self.n/len(self.srs))
        remaining = self.n
        for i, sr in enumerate(self.srs):
            if m is None:
                yield from sr
                continue
            if i == len(self.srs) - 1:
                m = remaining
            j = -1
            for j, line in zip(range(m), sr):
                yield line
            remaining -= j + 1

    def __getitem__(self, item):
        if item.start is not None or item.stop < 0 or item.step is not None:
            raise ValueError('only slicing from the start with step 1 is supported')
        return BalancedSlice(self.srs, item.stop if self.n is None else min(item.stop, self.n))


import json
import os
from contextlib import suppress


class Stream:
    def __init__(self, name):
        self.name = name


class PersistableStream(Stream):
    def __init__(self, name, src_srs=(), reuse=False, reuse_options=None):
        super().__init__(name)
        self.src_srs = src_srs
        self.reuse = reuse
        self.reuse_name = '{}.json'.format(name)
        self.reuse_options = reuse_options
        self.options_name = '{}.options.json'.format(name)

    def reusable(self):
        with suppress(AttributeError):
            if any(src_sr.reuse and not src_sr.reusable() for src_sr in self.src_srs):
                return False
        with suppress(FileNotFoundError), open(self.options_name) as sr:
            if self.reuse_options == json.load(sr):
                return True
        return False

    def _iter(self):
        raise NotImplementedError

    def __iter__(self):
        if self.reuse:
            if not self.reusable():
                with open(self.reuse_name, 'w') as sr:
                    for obj in self._iter():
                        sr.write(json.dumps(obj) + '\n')
                # only write options on success
                with open(self.options_name, 'w') as sr:
                    json.dump(self.reuse_options, sr)
            with open(self.reuse_name) as sr:
                for line in sr:
                    yield json.loads(line)
        else:
            yield from self._iter()


class SourceStream(Stream):
    def __init__(self, path):
        super().__init__(os.path.splitext(os.path.basename(path))[0])
        self.path = path

    def __iter__(self):
        with open(self.path) as sr:
            for line in sr:
                yield json.loads(line)


class MergedStream(Stream):
    def __init__(self, src_srs):
        super().__init__('_'.join(stream.name for stream in src_srs))
        self.src_srs = src_srs

    def __iter__(self):
        for sr in self.src_srs:
            yield from sr

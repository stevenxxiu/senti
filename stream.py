
import json
import os
from contextlib import suppress


class Stream:
    def __init__(self, name):
        self.name = name


class PersistableStream(Stream):
    def __init__(self, name, src_sr, reuse=False, options=None):
        super().__init__(name)
        self.src_sr = src_sr
        self.reuse = reuse
        self.reuse_name = '{}.json'.format(name)
        self.options = options
        self.options_name = '{}.options.json'.format(name)

    def can_reuse(self):
        with suppress(AttributeError):
            if self.src_sr.reuse and not self.src_sr.can_reuse():
                return False
        with suppress(FileNotFoundError), open(self.options_name) as sr:
            if self.options == json.load(sr):
                return True
        return False

    def _iter(self):
        raise NotImplementedError

    def __iter__(self):
        if self.reuse:
            if not self.can_reuse():
                with open(self.reuse_name, 'w') as sr:
                    for obj in self._iter():
                        sr.write(json.dumps(obj) + '\n')
                # only write options on success
                with open(self.options_name, 'w') as sr:
                    json.dump(self.options, sr)
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
    def __init__(self, streams):
        super().__init__('_'.join(stream.name for stream in streams))
        self.streams = streams

    def __iter__(self):
        for stream in self.streams:
            i = 0
            for i, obj in enumerate(stream):
                yield obj

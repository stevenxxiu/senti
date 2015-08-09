
import json
import os
from contextlib import suppress

from senti.utils import SciPyJSONEncoder, decode_scipy_object


class Stream:
    def __init__(self, name):
        self.name = name


class PersistableStream(Stream):
    def __init__(self, name, sr, src_srs=(), reuse=False, reuse_options=None):
        super().__init__(name)
        self.sr = sr
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

    def __iter__(self):
        if self.reuse:
            if not self.reusable():
                with open(self.reuse_name, 'w') as sr:
                    for obj in self.sr:
                        sr.write(json.dumps(obj, cls=SciPyJSONEncoder) + '\n')
                # only write options on success
                with open(self.options_name, 'w') as sr:
                    json.dump(self.reuse_options, sr)
            with open(self.reuse_name) as sr:
                for line in sr:
                    yield json.loads(line, object_hook=decode_scipy_object)
        else:
            yield from self.sr


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
        super().__init__('merge({})'.format(','.join(src_sr.name for src_sr in src_srs)))
        self.src_srs = src_srs

    def __iter__(self):
        for sr in self.src_srs:
            yield from sr


class SplitStream(Stream):
    def __init__(self, name, merged_sr_iter, src_sr):
        super().__init__(name)
        self.merged_sr_iter = merged_sr_iter
        self.src_sr = src_sr

    def __iter__(self):
        for src_obj, obj in zip(self.src_sr, self.merged_sr_iter):
            src_obj.update(obj)
            yield src_obj


def split_streams(merged_sr, src_srs):
    merged_sr_iter = iter(merged_sr)
    for src_sr in src_srs:
        yield SplitStream('{}[{}]'.format(merged_sr.name, src_sr.name), merged_sr_iter, src_sr)


import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np

from senti.stream import PersistableStream
from senti.utils import third_dir

__all__ = ['Word2Vec', 'Word2VecDocs', 'Word2VecWordAverage', 'Word2VecWordMax', 'Word2VecInverse']


class Word2Vec(PersistableStream):
    '''
    Performs a word embedding.
    '''

    def __init__(self, src_sr, cmd=None, reuse=False):
        self.cmd = cmd or '''
            -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1
            -sentence-vectors 1
        '''.replace('\n', '')
        super().__init__('{}.word2vec'.format(src_sr.name), (src_sr,), reuse, self.cmd)
        self.reuse_name = '{}.txt'.format(self.name)

    def _iter(self):
        with open('word2vec_train.txt', 'w', encoding='utf-8') as sr:
            for obj in self.src_srs[0]:
                sr.write('_*{} {}\n'.format(obj['id'], obj['text']))
        os.system(
            'time {}/word2vec/word2vec -train word2vec_train.txt -output {} {}'
            .format(third_dir, self.reuse_name, self.cmd)
        )

    def __iter__(self):
        if not self.reusable():
            self._iter()
            with open(self.options_name, 'w') as sr:
                json.dump(self.reuse_options, sr)
        with open(self.reuse_name, encoding='utf-8') as sr:
            yield from sr


class Word2VecDocs(PersistableStream):
    def __init__(self, src_sr, cmd=None, reuse=False):
        self.w2v_sr = Word2Vec(src_sr, cmd, reuse)
        super().__init__('{}.word2vec_docs'.format(src_sr.name), (src_sr,), reuse, self.w2v_sr.reuse_options)

    def _iter(self):
        for line in self.w2v_sr:
            if line.startswith('_*'):
                id_, vec = re.match(r'(\S+) (.+)', line).groups()
                yield {'id': id_, 'vec': list(map(float, vec.strip().split()))}


# noinspection PyAbstractClass
class Word2VecWords(PersistableStream):
    def __init__(self, name, src_sr, cmd=None, reuse=False):
        self.w2v_sr = Word2Vec(src_sr, cmd, reuse)
        super().__init__(name, (src_sr,), reuse, self.w2v_sr.reuse_options)

    def get_words(self):
        word_to_vecs = {}
        lines = iter(self.w2v_sr)
        dims = tuple(map(int, re.match(r'(\d+) (\d+)', next(lines)).groups()))
        for line in lines:
            word, vec = re.match(r'(\S+) (.+)', line).groups()
            word_to_vecs[word] = np.array(list(map(float, vec.strip().split())))
        return dims, word_to_vecs


class Word2VecWordAverage(Word2VecWords):
    def __init__(self, src_sr, cmd=None, reuse=False):
        super().__init__('{}.word2vec_avg'.format(src_sr.name), src_sr, cmd, reuse)

    def _iter(self):
        dims, word_to_vecs = self.get_words()
        for obj in self.src_srs[0]:
            words = obj['text'].split()
            vec = np.vstack(word_to_vecs[word] for word in words if word in word_to_vecs).mean(0)
            yield {'id': obj['id'], 'vec': vec.tolist()}


class Word2VecWordMax(Word2VecWords):
    '''
    Component-wise abs max. Doesn't make much sense because the components aren't importance, but worth a try.
    '''

    def __init__(self, src_sr, cmd=None, reuse=False):
        super().__init__('{}.word2vec_max'.format(src_sr.name), src_sr, cmd, reuse)

    def _iter(self):
        dims, word_to_vecs = self.get_words()
        for obj in self.src_srs[0]:
            words = obj['text'].split()
            words_matrix = np.vstack(word_to_vecs[word] for word in words if word in word_to_vecs)
            arg_maxes = abs(words_matrix).argmax(0)
            vec = words_matrix[arg_maxes, np.arange(len(arg_maxes))]
            yield {'id': obj['id'], 'vec': vec.tolist()}


class Word2VecInverse(PersistableStream):
    '''
    Performs a document embedding.
    '''

    def __init__(self, src_sr, cmd=None, reuse=False):
        self.cmd = cmd or '''
            -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1
            -sentence-vectors 1
        '''.replace('\n', '')
        super().__init__('{}.word2vec_inv'.format(src_sr.name), (src_sr,), reuse, self.cmd)
        self.reuse_name = '{}.txt'.format(self.name)

    def _iter(self):
        # number documents using integers for easy sorting later
        word_to_docs = defaultdict(set)
        for i, obj in enumerate(self.src_srs[0]):
            words = obj['text'].split()
            for word in words:
                word_to_docs[word].add(str(i))
        with open('word2vec_train.txt', 'w', encoding='utf-8') as out_sr:
            for word, docs in word_to_docs.items():
                # remove common & rare words
                if 10 <= len(docs) <= 2000:
                    out_sr.write('_*{} {}\n'.format(word, ' '.join(docs)))
        os.system(
            'time {}/word2vec/word2vec -train word2vec_train.txt -output {} {}'
            .format(third_dir, self.reuse_name, self.cmd)
        )
        with open(self.reuse_name, encoding='ISO-8859-1') as in_sr, open('{}~'.format(self.reuse_name), 'w') as out_sr:
            for line in itertools.islice(in_sr, 1, None):
                if not line.startswith('_*') and not line.startswith('</s>'):
                    out_sr.write(line)
        os.system('sort -n {0}~ > {0}'.format(self.reuse_name))
        os.system('rm {}~'.format(self.reuse_name))

    def __iter__(self):
        if not self.reusable():
            self._iter()
            with open(self.options_name, 'w') as sr:
                json.dump(self.reuse_options, sr)
        with open(self.reuse_name) as sr:
            for obj, line in zip(self.src_srs[0], sr):
                vec = re.match(r'(\S+) (.+)', line).group(2)
                yield {'id': obj['id'], 'vec': list(map(float, vec.strip().split()))}


import itertools
import os
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Word2Vec', 'Doc2Vec', 'Doc2VecTransform', 'Word2VecAverage', 'Word2VecMax', 'Word2VecInverse']


class Word2VecBase:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        os.system('../../third/word2vec/word2vec -train sentences.txt -output sentence_vecs.txt {}'.format(
            ' '.join('-{} {}'.format(key.replace('_', '-'), val) for key, val in self.kwargs.items())
        ))
        print()


class Word2Vec(Word2VecBase):
    def __init__(self, **kwargs):
        super().__init__(sentence_vectors=0, **kwargs)
        self.words = {}
        self.X = None

    def fit(self, docs):
        with open('sentences.txt', 'w', encoding='utf-8') as sr:
            for doc in docs:
                sr.write('{}\n'.format(' '.join(doc)))
        self.run()
        with open('sentence_vecs.txt', encoding='ISO-8859-1') as sr:
            vecs = []
            for i, line in enumerate(itertools.islice(sr, 1, None)):
                parts = line.split(' ')
                self.words[parts[0]] = i
                vecs.append(np.fromiter(parts[1:-1], np.float64))
            self.X = np.vstack(vecs)
        return self

    def load_binary(self, name):
        with open(name, 'rb') as sr:
            header = sr.readline()
            vocab_size, layer1_size = tuple(map(int, header.split()))
            binary_len = np.float32().itemsize*layer1_size
            vecs = []
            for i in range(vocab_size):
                word = bytearray()
                while True:
                    ch = sr.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':
                        # ignore newlines in front of words (some binary files have them)
                        word.append(ch[0])
                self.words[word.decode('utf-8')] = i
                vecs.append(np.frombuffer(sr.read(binary_len), dtype=np.float32))
            self.X = np.vstack(vecs)


class Doc2Vec(Word2VecBase):
    def __init__(self, **kwargs):
        super().__init__(sentence_vectors=1, **kwargs)
        self.X = None

    def fit(self, docs):
        with open('sentences.txt', 'w', encoding='utf-8') as sr:
            for i, doc in enumerate(docs):
                sr.write('_*{} {}\n'.format(i, ' '.join(doc)))
        self.run()
        with open('sentence_vecs.txt', encoding='ISO-8859-1') as sr:
            vecs = []
            for line in itertools.islice(sr, 1, None):
                if line.startswith('_*'):
                    vecs.append(np.fromiter(line.split()[1:], np.float64))
            self.X = np.vstack(vecs)
        return self


class Doc2VecTransform(BaseEstimator):
    def __init__(self, dev_docs, unsup_docs=(), **kwargs):
        self.docs = {'train': (), 'unsup': unsup_docs, 'dev': dev_docs}
        self._docs_start = {}
        self._docs_end = {}
        self.word2vec = Doc2Vec(**kwargs)

    def _all_docs(self):
        pos = 0
        for name in ('train', 'unsup', 'dev'):
            self._docs_start[name] = pos
            i = -1
            for i, doc in enumerate(self.docs[name]):
                yield doc
            pos += i + 1
            self._docs_end[name] = pos

    def fit(self, docs, y=None):
        self.docs['train'] = docs
        self.word2vec.fit(self._all_docs())
        return self

    def transform(self, docs):
        for name in ('train', 'dev'):
            if docs == self.docs[name]:
                return self.word2vec.X[range(self._docs_start[name], self._docs_end[name])]
        raise ValueError('docs were not fitted')


class Word2VecTransform(BaseEstimator):
    def __init__(self, unsup_docs=(), pretrained_file=None, **kwargs):
        self.unsup_docs = unsup_docs
        self.word2vec = Word2Vec(**kwargs)
        self.pretrained = bool(pretrained_file)
        if pretrained_file:
            self.word2vec.load_binary(pretrained_file)

    def fit(self, docs, y=None):
        if not self.pretrained:
            self.word2vec.fit(itertools.chain(docs, self.unsup_docs))


class Word2VecAverage(Word2VecTransform):
    def transform(self, docs):
        vecs = []
        words, X = self.word2vec.words, self.word2vec.X
        for doc in docs:
            vecs.append(np.mean(np.vstack(X[words[word]] for word in doc if word in words), axis=0))
        return np.vstack(vecs)


class Word2VecMax(Word2VecTransform):
    '''
    Component-wise abs max. Doesn't make much sense because the components aren't importance, but worth a try.
    '''

    def transform(self, docs):
        vecs = []
        words, X = self.word2vec.words, self.word2vec.X
        for doc in docs:
            words_matrix = np.vstack(X[words[word]] for word in doc if word in words)
            arg_maxes = np.abs(words_matrix).argmax(0)
            vecs.append(words_matrix[arg_maxes, np.arange(len(arg_maxes))])
        return np.vstack(vecs)


class Word2VecInverse(BaseEstimator):
    '''
    Performs a document embedding.
    '''

    def __init__(self, dev_docs, unsup_docs=(), docs_min=10, docs_max=2000, **kwargs):
        self.docs = {'train': (), 'unsup': unsup_docs, 'dev': dev_docs}
        self.docs_min = docs_min
        self.docs_max = docs_max
        self._docs_start = {}
        self._docs_end = {}
        self.word2vec = Word2Vec(**kwargs)

    def _all_docs(self):
        pos = 0
        for name in ('train', 'unsup', 'dev'):
            self._docs_start[name] = pos
            i = -1
            for i, doc in enumerate(self.docs[name]):
                yield doc
            pos += i + 1
            self._docs_end[name] = pos

    def fit(self, docs, y=None):
        self.docs['train'] = docs
        # number documents using integers for easy sorting later
        word_to_docs = defaultdict(set)
        for i, words in enumerate(self._all_docs()):
            for word in words:
                word_to_docs[word].add(str(i))
        # remove common & rare words
        self.word2vec.fit((doc for doc in word_to_docs.values() if self.docs_min <= len(doc) <= self.docs_max))
        return self

    def transform(self, docs):
        for name in ('train', 'dev'):
            if docs == self.docs[name]:
                return self.word2vec.X[list(
                    self.word2vec.words[str(i)] for i in range(self._docs_start[name], self._docs_end[name])
                )]
        raise ValueError('docs were not fitted')

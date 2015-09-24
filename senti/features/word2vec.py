
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
        self.word_to_index = {}
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
                self.word_to_index[parts[0]] = i
                vecs.append(np.fromiter(parts[1:-1], np.float32))
            self.X = np.vstack(vecs)
        return self


class Doc2Vec(Word2VecBase):
    def __init__(self, **kwargs):
        super().__init__(sentence_vectors=1, **kwargs)
        self.X = None
        self.prefix = '_*/'

    def fit(self, docs):
        with open('sentences.txt', 'w', encoding='utf-8') as sr:
            for i, doc in enumerate(docs):
                sr.write('{}{} {}\n'.format(self.prefix, i, ' '.join(doc)))
        self.run()
        with open('sentence_vecs.txt', encoding='ISO-8859-1') as sr:
            vecs = []
            for line in itertools.islice(sr, 1, None):
                if line.startswith(self.prefix):
                    vecs.append(np.fromiter(line.split()[1:], np.float32))
            self.X = np.vstack(vecs)
        return self


class Word2VecFeatureBase(BaseEstimator):
    def __init__(self, unsup_docs=(), word2vec=None, **kwargs):
        self.unsup_docs = unsup_docs
        self.word2vec = word2vec or Word2Vec(**kwargs)

    def fit(self, docs, y=None):
        if not self.word2vec.word_to_index:
            self.word2vec.fit(itertools.chain(docs, self.unsup_docs))


class Word2VecAverage(Word2VecFeatureBase):
    def transform(self, docs):
        vecs = []
        words, X = self.word2vec.word_to_index, self.word2vec.X
        for doc in docs:
            vecs.append(np.mean(np.vstack(X[words[word]] for word in doc if word in words), axis=0))
        return np.vstack(vecs)


class Word2VecMax(Word2VecFeatureBase):
    '''
    Component-wise abs max. Doesn't make much sense because the components aren't importance, but worth a try.
    '''

    def transform(self, docs):
        vecs = []
        words, X = self.word2vec.word_to_index, self.word2vec.X
        for doc in docs:
            words_matrix = np.vstack(X[words[word]] for word in doc if word in words)
            arg_maxes = np.abs(words_matrix).argmax(0)
            vecs.append(words_matrix[arg_maxes, np.arange(len(arg_maxes))])
        return np.vstack(vecs)


class Doc2VecFeatureBase(BaseEstimator):
    def __init__(self, corporas=()):
        self.corporas = corporas
        self._corporas_start = []
        self._corporas_end = []

    def _all_docs(self):
        pos = 0
        for corpora in self.corporas:
            self._corporas_start.append(pos)
            i = -1
            for i, doc in enumerate(corpora):
                yield doc
            pos += i + 1
            self._corporas_end.append(pos)


class Doc2VecTransform(Doc2VecFeatureBase):
    def __init__(self, corporas=(), **kwargs):
        super().__init__(corporas)
        self.word2vec = Doc2Vec(**kwargs)

    def fit(self, docs, y=None):
        self.corporas.append(docs)
        self.word2vec.fit(self._all_docs())
        return self

    def transform(self, docs):
        try:
            i = self.corporas.index(docs)
        except IndexError:
            raise ValueError('docs were not fitted')
        return self.word2vec.X[self._corporas_start[i]:self._corporas_end[i]]


class Word2VecInverse(Doc2VecFeatureBase):
    '''
    Performs a document embedding.
    '''

    def __init__(self, corporas=(), docs_min=10, docs_max=2000, **kwargs):
        super().__init__(corporas)
        self.docs_min = docs_min
        self.docs_max = docs_max
        self.word2vec = Word2Vec(**kwargs)

    def fit(self, docs, y=None):
        self.corporas.append(docs)
        # number documents using integers for easy sorting later
        word_to_docs = defaultdict(set)
        for i, words in enumerate(self._all_docs()):
            for word in words:
                word_to_docs[word].add(str(i))
        # remove common & rare words
        self.word2vec.fit((doc for doc in word_to_docs.values() if self.docs_min <= len(doc) <= self.docs_max))
        return self

    def transform(self, docs):
        try:
            i = self.corporas.index(docs)
        except IndexError:
            raise ValueError('docs were not fitted')
        return self.word2vec.X[list(
            self.word2vec.word_to_index[str(j)] for j in range(self._corporas_start[i], self._corporas_end[i])
        )]

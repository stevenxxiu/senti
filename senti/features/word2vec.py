
import itertools
import os
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator

__all__ = ['Word2Vec', 'Doc2Vec', 'Word2VecAverage', 'Word2VecMax', 'Doc2VecTransform', 'Word2VecInverse']


class Word2VecBase(BaseEstimator):
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

    def fit(self, docs, rand=None):
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


class FeatureBase(BaseEstimator):
    def __init__(self, word2vec):
        self.word2vec = word2vec

    def get_params(self, deep=True):
        res = super().get_params(deep)
        res['word2vec'] = self.word2vec.kwargs
        return res

    def fit(self, docs, y=None):
        if not getattr(self.word2vec, 'word_to_index', None):
            self.word2vec.fit(docs)


class Word2VecAverage(FeatureBase):
    def transform(self, docs):
        vecs = []
        words, X = self.word2vec.word_to_index, self.word2vec.X
        for doc in docs:
            vecs.append(np.mean(np.vstack(X[words[word]] for word in doc if word in words), axis=0))
        return np.vstack(vecs)


class Word2VecMax(FeatureBase):
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


class Doc2VecTransform(FeatureBase):
    def transform(self, docs):
        return self.word2vec.X


class Word2VecInverse(FeatureBase):
    '''
    Performs a document embedding.
    '''

    def __init__(self, word2vec, docs_min=10, docs_max=2000):
        super().__init__(word2vec)
        self.docs_min = docs_min
        self.docs_max = docs_max

    def fit(self, docs, y=None):
        # number documents using integers for easy sorting later
        word_to_docs = defaultdict(set)
        for i, words in docs:
            for word in words:
                word_to_docs[word].add(str(i))
        # remove common & rare words
        self.word2vec.fit((doc for doc in word_to_docs.values() if self.docs_min <= len(doc) <= self.docs_max))
        return self

    def transform(self, docs):
        return self.word2vec.X[list(
            self.word2vec.word_to_index[str(i)] for i in range(len(self.word2vec.word_to_index))
        )]

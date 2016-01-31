
from collections import defaultdict

import numpy as np
from gensim.models.doc2vec import TaggedDocument
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize

__all__ = ['Word2VecAverage', 'Word2VecNormAverage', 'Word2VecMax', 'Word2VecInverse', 'Doc2VecTransform']


class Word2VecBase(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, docs, y=None):
        if not hasattr(self.model, 'syn0'):
            self.model.build_vocab(docs)
            self.model.fit(docs)
        return self


class Word2VecAverage(Word2VecBase):
    def transform(self, docs):
        vecs = []
        for doc in docs:
            try:
                words_matrix = np.vstack(self.model[word] for word in doc if word in self.model.vocab)
                vecs.append(np.mean(words_matrix, axis=0))
            except ValueError:
                vecs.append(np.zeros(self.model.vector_size))
        return np.vstack(vecs)


class Word2VecNormAverage(Word2VecBase):
    def transform(self, docs):
        vecs = []
        for doc in docs:
            try:
                words_matrix = np.vstack(self.model[word] for word in doc if word in self.model.vocab)
                vecs.append(normalize(np.mean(normalize(words_matrix), axis=0)))
            except ValueError:
                vecs.append(np.zeros(self.model.vector_size))
        return np.vstack(vecs)


class Word2VecMax(Word2VecBase):
    def transform(self, docs):
        vecs = []
        for doc in docs:
            try:
                words_matrix = np.vstack(self.model[word] for word in doc if word in self.model.vocab)
                arg_maxes = np.abs(words_matrix).argmax(0)
                vecs.append(words_matrix[arg_maxes, np.arange(len(arg_maxes))])
            except ValueError:
                vecs.append(np.zeros(self.model.vector_size))
        return np.vstack(vecs)


class Word2VecInverse(Word2VecBase):
    '''
    Embeds documents by transposing the term-document matrix (letting documents be words).
    '''

    def __init__(self, model):
        super().__init__(model)

    def fit(self, docs, y=None):
        # number documents using integers
        word_to_docs = defaultdict(set)
        for i, words in enumerate(docs):
            for word in words:
                word_to_docs[word].add(str(i))
        word_docs = list(map(list, word_to_docs.values()))
        self.model.build_vocab(word_docs)
        self.model.fit(word_docs)
        return self

    def transform(self, docs):
        return self.model[map(str, range(len(self.model.vocab)))]


class Doc2VecTransform(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, docs, y=None):
        docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
        self.model.build_vocab(docs)
        self.model.fit(docs)
        return self

    def transform(self, docs):
        return self.model.docvecs.doctag_syn0

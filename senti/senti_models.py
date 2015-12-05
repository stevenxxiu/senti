
import logging
import re
from types import SimpleNamespace

import joblib
import numpy as np
from joblib import Memory
from scipy.optimize import basinhopping
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, LabelEncoder
from sklearn.svm import LinearSVC

from senti.features import *
from senti.models import *
from senti.preprocess import *
from senti.rand import get_rng
from senti.transforms import *
from senti.utils import compose, temp_log_level
from senti.utils.cache import CachedFitTransform
from senti.utils.gensim_ import *

__all__ = ['SentiModels']


class LazyLabels:
    def __init__(self, labels):
        self.labels = labels
        self.value = None

    def __getitem__(self, item):
        return type(self)(self.labels[item])

    def __call__(self):
        if self.value is None:
            self.value = np.fromiter(self.labels, 'int32')
        return self.value


class SentiModels:
    def __init__(
        self, unsup_docs, distant_docs, distant_labels, train_docs, train_labels, dev_docs, dev_labels, test_docs
    ):
        self.unsup_docs = unsup_docs
        self.distant_docs = distant_docs
        self.distant_labels = LazyLabels(distant_labels)
        self.train_docs = train_docs
        self.train_labels = LazyLabels(train_labels)
        self.dev_docs = dev_docs
        self.dev_labels = LazyLabels(dev_labels)
        self.test_docs = test_docs
        self.memory = Memory(cachedir='cache', verbose=0)

    def fit_voting(self):
        names = [
            # 'svm(word_n_grams,char_n_grams,all_caps,hashtags,punctuations,punctuation_last,emoticons,emoticon_last,'
            # 'elongated,negation_count)',
            # 'logreg(w2v_doc)',
            'logreg(w2v_word_avg_google)',
            'word2vec_bayes',
            'cnn(embedding=google)',
        ]
        classifiers = [ExternalModel({
            self.dev_docs: 'results/dev/{}.json'.format(name), self.test_docs: 'results/test/{}.json'.format(name)
        }) for name in names]
        all_probs = np.array([classifier.predict_proba(self.dev_docs) for classifier in classifiers])
        all_probs_first, all_probs_rest = all_probs[0], all_probs[1:]
        label_encoder = LabelEncoder()
        dev_label_indexes = label_encoder.fit_transform(self.dev_labels())
        # assume w_0=1 as w is invariant to scaling
        w = basinhopping(
            lambda w_: -(dev_label_indexes == np.argmax((
                all_probs_first + all_probs_rest*w_.reshape((len(w_), 1, 1))
            ).sum(axis=0), axis=1)).sum(), get_rng().uniform(0, 1, len(classifiers) - 1), niter=1000,
            minimizer_kwargs=dict(method='L-BFGS-B', bounds=[(0, None)]*(len(classifiers) - 1))
        ).x
        w = np.hstack([[1], w])
        w /= w.sum()
        print('w: {}'.format(w))
        estimator = VotingClassifier(list(zip(names, classifiers)), voting='soft', weights=w)
        estimator.classes_ = label_encoder.classes_
        estimator.estimators_ = classifiers
        return 'vote({})'.format(','.join(names)), estimator

    def fit_svm(self):
        tokenize_raw = CachedFitTransform(Map(compose(tokenize, normalize_special, unescape)), self.memory)
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', tokenize_raw), ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        tokenize_insense = CachedFitTransform(Pipeline([
            ('tokenize', tokenize_sense), ('normalize', MapTokens(str.lower)),
        ]), self.memory)
        features = FeatureUnion([
            ('word_n_grams', FeatureUnion([(n, CachedFitTransform(Pipeline([
                ('tokenize', tokenize_insense),
                ('negation_append', NegationAppend()),
                ('ngrams', WordNGrams(n)),
                ('count', Count()),
                ('binarize', Binarizer(copy=False)),
            ]), self.memory)) for n in range(1, 4 + 1)])),
            ('char_n_grams', FeatureUnion([(n, CachedFitTransform(Pipeline([
                ('tokenize', tokenize_insense),
                ('ngrams', CharNGrams(n)),
                ('count', Count()),
                ('binarize', Binarizer(copy=False)),
            ]), self.memory)) for n in range(3, 5 + 1)])),
            ('all_caps', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', AllCaps()),
                ('count', Count()),
            ])),
            # XXX pos
            ('hashtags', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', HashTags()),
                ('count', Count()),
            ])),
            # XXX lexicons
            ('punctuations', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Punctuations()),
                ('count', Count()),
            ])),
            ('punctuation_last', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Punctuations()),
                ('last', Index(-1)),
            ])),
            ('emoticons', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Emoticons()),
                ('count', Count()),
                ('binarize', Binarizer(copy=False)),
            ])),
            ('emoticon_last', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Emoticons()),
                ('last', Index(-1)),
            ])),
            ('elongated', Pipeline([
                ('tokenize', tokenize_raw),
                ('feature', Elongations()),
                ('count', Count()),
            ])),
            ('negation_count', Pipeline([
                ('tokenize', tokenize_insense),
                ('feature', NegationCount()),
            ])),
        ])
        estimator = Pipeline([('features', features), ('classifier', LinearSVC(C=0.005))])
        estimator.fit(self.train_docs, self.train_labels())
        return 'svm({})'.format(','.join(name for name, _ in features.transformer_list)), estimator

    def fit_logreg(self):
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special, unescape))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        features = FeatureUnion([
            # ('w2v_doc', AsCorporas(Pipeline([
            #     ('tokenize', MapCorporas(tokenize_sense)),
            #     ('feature', MergeSliceCorporas(Doc2VecTransform(CachedFitTransform(Doc2Vec(
            #         dm=0, dbow_words=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20,
            #         workers=16
            #     ), self.memory)))),
            # ]).fit([self.train_docs, self.unsup_docs[:10**6], self.dev_docs, self.test_docs]))),
            # ('w2v_word_avg', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecAverage(CachedFitTransform(Word2Vec(
            #         sg=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
            #     ), self.memory))),
            # ]).fit(self.unsup_docs[:10**6])),
            # ('w2v_word_avg_google', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecAverage(joblib.load('../google/GoogleNews-vectors-negative300.pickle'))),
            # ])),
            # ('w2v_word_norm_avg', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecNormAverage(CachedFitTransform(Word2Vec(
            #         sg=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
            #     ), self.memory))),
            # ]).fit(self.unsup_docs[:10**6])),
            ('w2v_word_norm_avg_google', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Word2VecNormAverage(joblib.load('../google/GoogleNews-vectors-negative300.pickle'))),
            ])),
            # ('w2v_word_max', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecMax(CachedFitTransform(Word2Vec(
            #         sg=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
            #     ), self.memory))),
            # ]).fit(self.unsup_docs[:10**6])),
            # ('w2v_word_max_google', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecMax(joblib.load('../google/GoogleNews-vectors-negative300.pickle'))),
            # ])),
            # ('w2v_word_inv', AsCorporas(Pipeline([
            #     ('tokenize', MapCorporas(tokenize_sense)),
            #     ('feature', MergeSliceCorporas(Word2VecInverse(CachedFitTransform(Word2Vec(
            #         sg=1, size=100, window=10, hs=0, negative=5, sample=0, min_count=1, iter=20, workers=16
            #     ), self.memory)))),
            # ]).fit([self.train_docs, self.unsup_docs[:10**5], self.dev_docs, self.test_docs]))),
        ])
        classifier = LogisticRegression()
        with temp_log_level({'gensim.models.word2vec': logging.INFO}):
            classifier.fit(features.transform(self.train_docs), self.train_labels())
        estimator = Pipeline([('features', features), ('classifier', classifier)])
        return 'logreg({})'.format(','.join(name for name, _ in features.transformer_list)), estimator

    def fit_word2vec_bayes(self):
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special, unescape))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        estimator = Pipeline([
            ('tokenize', tokenize_sense),
            ('classifier', Word2VecBayes(Word2Vec(
                sg=1, size=100, window=10, hs=1, sample=0, min_count=5, workers=16
            )))
        ])
        with temp_log_level({'senti.models.word2vec_bayes': logging.INFO, 'gensim.models.word2vec': logging.ERROR}):
            estimator.fit(self.train_docs, self.train_labels())
        return 'word2vec_bayes', estimator

    def fit_embedding(self, embedding_type, construct_docs, tokenize_):
        if embedding_type == 'google':
            embeddings_ = joblib.load('../google/GoogleNews-vectors-negative300.pickle')
            embeddings_ = SimpleNamespace(X=embeddings_.syn0, vocab={w: v.index for w, v in embeddings_.vocab.items()})
        elif embedding_type == 'twitter':
            estimator = Pipeline([
                ('tokenize', MapCorporas(tokenize_)),
                ('word2vec', MergeSliceCorporas(CachedFitTransform(Word2Vec(
                    sg=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
                ), self.memory)))
            ]).fit([self.train_docs, self.unsup_docs[:10**6], self.dev_docs, self.test_docs])
            embeddings_ = estimator.named_steps['word2vec'].estimator
            embeddings_ = SimpleNamespace(X=embeddings_.syn0, vocab={w: v.index for w, v in embeddings_.vocab.items()})
        else:
            embeddings_ = SimpleNamespace(X=np.empty((0, 100)), vocab={})
        estimator = Pipeline([
            ('tokenize', tokenize_),
            # 0.25 is chosen so the unknown vectors have approximately the same variance as google pre-trained ones
            ('embeddings', Embeddings(
                embeddings_, rand=lambda shape: get_rng().uniform(-0.25, 0.25, shape), include_zero=True
            ))
        ])
        for docs in construct_docs:
            estimator.fit(docs)
        return estimator, estimator.named_steps['embeddings']

    def fit_cnn(self):
        embedding_type = 'google'
        construct_docs = [self.dev_docs, self.train_docs]
        distant_docs, distant_labels = self.distant_docs[:10**5], self.distant_labels[:10**5]
        construct_docs.append(distant_docs)
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        features, embeddings_ = self.fit_embedding(embedding_type, construct_docs, tokenize_sense)
        classifier = CNN(
            batch_size=64, embeddings=embeddings_, input_size=56, conv_param=(100, [3, 4, 5]), dense_params=[],
            output_size=3, static_mode=1, norm_lim=3
        )
        features = Pipeline([('index', features), ('clip', Clip(56))])
        args = dict(dev_X=features.transform(self.dev_docs), dev_y=self.dev_labels(), average_classes=[0, 2])
        classifier.fit(features.transform(distant_docs), distant_labels(), max_epochs=1, **args)
        classifier.fit(features.transform(self.train_docs), self.train_labels(), max_epochs=10, **args)
        estimator = Pipeline([('features', features), ('classifier', classifier)])
        return 'cnn(embedding={})'.format(embedding_type), estimator

    def fit_cnn_char(self):
        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
        features = Pipeline([
            ('tokenize', Map(compose(str.lower, str.strip, lambda s: re.sub(r'\s+', ' ', s), normalize_special))),
            ('embeddings', Embeddings(SimpleNamespace(
                vocab=dict(zip(alphabet, range(len(alphabet)))), X=np.identity(len(alphabet), dtype='float32')
            ), include_zero=True))
        ])
        classifier = CNNChar(batch_size=128, embeddings=features.named_steps['embeddings'], input_size=140)
        args = dict(dev_X=features.transform(self.dev_docs), dev_y=self.dev_labels(), average_classes=[0, 2])
        classifier.fit(features.transform(self.train_docs), self.train_labels(), epoch_len=20, max_epochs=100, **args)
        estimator = Pipeline([('features', features), ('classifier', classifier)])
        return 'cnn_char', estimator

    def fit_rnn(self):
        embedding_type = 'google'
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        features, embeddings_ = self.fit_embedding(embedding_type, [self.dev_docs, self.train_docs], tokenize_sense)
        classifier = RNN(batch_size=64, embeddings=embeddings_, lstm_param=300, output_size=3)
        args = dict(dev_X=features.transform(self.dev_docs), dev_y=self.dev_labels(), average_classes=[0, 2])
        classifier.fit(features.transform(self.train_docs), self.train_labels(), epoch_len=20, max_epochs=100, **args)
        estimator = Pipeline([('features', features), ('classifier', classifier)])
        return 'rnn(embedding={})'.format(embedding_type), estimator

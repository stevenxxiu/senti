
import logging
from types import SimpleNamespace

import joblib
import numpy as np
from joblib import Memory
from lasagne.nonlinearities import identity, rectify
from scipy.optimize import basinhopping
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, LabelEncoder
from sklearn.svm import LinearSVC

from senti.cache import CachedFitTransform
from senti.features import *
from senti.gensim_ext import *
from senti.models import *
from senti.preprocess import *
from senti.rand import get_rng
from senti.transforms import *
from senti.utils import compose, temp_log_level

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
            'svm(word_n_grams,char_n_grams,all_caps,hashtags,punctuations,punctuation_last,emoticons,emoticon_last,'
            'elongated,negation_count)',
            'cnn(embedding=twitter)',
        ]
        classifiers = [ExternalModel({
            self.dev_docs: 'results/dev/{}.json'.format(name), self.test_docs: 'results/test/{}.json'.format(name)
        }) for name in names]
        classifier_probs = np.array([classifier.predict_proba(self.dev_docs) for classifier in classifiers])
        label_encoder = LabelEncoder()
        dev_label_indexes = label_encoder.fit_transform(self.dev_labels())
        # assume w_0=1 as w is invariant to scaling
        w = basinhopping(
            lambda w_: -(
                dev_label_indexes == np.argmax((
                    classifier_probs*np.hstack([[1], w_]).reshape((len(w_) + 1, 1, 1))
                ).sum(axis=0), axis=1)
            ).sum(), get_rng().uniform(0, 1, len(classifiers) - 1),
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
        tokenize_raw = CachedFitTransform(Map(compose([tokenize, normalize_special, unescape])), self.memory)
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
            ('tokenize', Map(compose([tokenize, normalize_special, unescape]))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        features = FeatureUnion([
            # ('w2v_doc', AsCorporas(Pipeline([
            #     ('tokenize', MapCorporas(tokenize_sense)),
            #     ('feature', MergeSliceCorporas(Doc2VecTransform(CachedFitTransform(Doc2Vec(
            #         dm=0, dbow_words=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20,
            #         workers=16, batch_target=10000
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
            ('tokenize', Map(compose([tokenize, normalize_special, unescape]))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        estimator = Pipeline([
            ('tokenize', tokenize_sense),
            ('classifier', Word2VecBayes(Word2Vec(workers=16)))
        ])
        with temp_log_level({'senti.models.word2vec_bayes': logging.INFO, 'gensim.models.word2vec': logging.ERROR}):
            estimator.fit(self.train_docs, self.train_labels())
        return 'word2vec_bayes', estimator

    def fit_cnn(self):
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose([tokenize, normalize_special]))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        embedding_type = 'google'
        if embedding_type == 'google':
            embeddings_ = joblib.load('../google/GoogleNews-vectors-negative300.pickle')
            embeddings_ = SimpleNamespace(X=embeddings_.syn0, vocab={w: v.index for w, v in embeddings_.vocab.items()})
        elif embedding_type == 'twitter':
            embeddings_ = Pipeline([
                ('tokenize', MapCorporas(tokenize_sense)),
                ('word2vec', MergeSliceCorporas(CachedFitTransform(Word2Vec(
                    sg=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
                ), self.memory)))
            ]).fit([self.train_docs, self.unsup_docs[:10**6], self.dev_docs, self.test_docs])
            embeddings_ = embeddings_.named_steps['word2vec'].estimator
            embeddings_ = SimpleNamespace(X=embeddings_.syn0, vocab={w: v.index for w, v in embeddings_.vocab.items()})
        else:
            embeddings_ = SimpleNamespace(X=np.empty((0, 300)), vocab={})
        features = Pipeline([
            ('tokenize', tokenize_sense),
            # 0.25 is chosen so the unknown vectors have approximately the same variance as google pre-trained ones
            ('index', EmbeddingConstructor(embeddings_, lambda shape: get_rng().uniform(-0.25, 0.25, shape), True)),
            ('clip', Clip(56))
        ])
        distant_docs, distant_labels = self.distant_docs[:10**5], self.distant_labels[:10**5]
        features.fit(distant_docs)
        features.fit(self.dev_docs)
        features.fit(self.train_docs)
        classifier = ConvNet(
            batch_size=50, embeddings=features.named_steps['index'], img_h=56, filter_hs=[3, 4, 5],
            hidden_units=[100, 3], dropout_rates=[0.5], conv_non_linear=rectify, activations=(identity,),
            static_mode=1, lr_decay=0.95, norm_lim=3
        )
        fit_args = dict(dev_X=features.transform(self.dev_docs), dev_y=self.dev_labels(), average_classes=[0, 2])
        classifier.fit(
            features.transform(distant_docs), distant_labels(), shuffle_batch=False, n_epochs=1, **fit_args
        )
        classifier.fit(
            features.transform(self.train_docs), self.train_labels(), shuffle_batch=True, n_epochs=16, **fit_args
        )
        estimator = Pipeline([('features', features), ('classifier', classifier)])
        return 'cnn(embedding={})'.format(embedding_type), estimator

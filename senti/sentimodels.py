
import joblib
import numpy as np
from joblib import Memory
from lasagne.nonlinearities import identity, rectify
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.svm import LinearSVC

from senti.cache import CachedFitTransform
from senti.features import *
from senti.models import *
from senti.preprocess import *
from senti.rand import get_rng
from senti.transforms import *
from senti.utils import compose

__all__ = ['SentiModels']


class SentiModels:
    def __init__(
        self, train_docs, train_labels, distant_docs, distant_labels, unsup_docs, dev_docs, dev_labels, test_docs
    ):
        self.train_docs = train_docs
        self.train_labels = train_labels
        self.distant_docs = distant_docs
        self.distant_labels = distant_labels
        self.unsup_docs = unsup_docs
        self.dev_docs = dev_docs
        self.test_docs = test_docs
        self.dev_labels = dev_labels
        self.memory = Memory(cachedir='cache', verbose=0)

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
        name = 'svm({})'.format(','.join(name for name, _ in features.transformer_list))
        estimator = Pipeline([('features', features), ('svm', LinearSVC(C=0.005))])
        estimator.fit(self.train_docs, self.train_labels)
        return name, estimator

    def fit_logreg(self):
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose([tokenize, normalize_special, unescape]))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        features = FeatureUnion([
            ('w2v_doc', CachedFitTransform(FixedTransformWrapper(Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Doc2VecTransform(
                    cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
                )),
            ])), self.memory).fit([self.train_docs, self.unsup_docs[:10**6], self.dev_docs, self.test_docs])),
            # ('w2v_word_avg', CachedFitTransform(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecAverage(
            #         cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
            #     )),
            # ]), self.memory).fit(self.unsup_docs[:10**6])),
            # ('w2v_word_avg_google', CachedFitTransform(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', (Word2VecAverage(
            #         word2vec=joblib.load('../google/GoogleNews-vectors-negative300.pickle')
            #     )))
            # ]), self.memory)),
            # ('w2v_word_max', CachedFitTransform(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecMax(
            #         cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
            #     ))
            # ]), self.memory).fit(self.unsup_docs[:10**6])),
            # ('w2v_word_inv', CachedFitTransform(FixedTransformWrapper(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecInverse(
            #         cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
            #     ))
            # ])), self.memory).fit([self.unsup_docs[:10**5], self.dev_docs, self.test_docs])),
        ])
        name = 'logreg({})'.format(','.join(name for name, _ in features.transformer_list))
        classifier = LogisticRegression()
        classifier.fit(features.transform(self.train_docs), np.fromiter(self.train_labels, 'int32'))
        estimator = Pipeline([('features', features), ('logreg', classifier)])
        return name, estimator

    def fit_cnn(self):
        use_w2v = True
        name = 'cnn(use_w2v={})'.format(use_w2v)
        feature = Pipeline([
            ('tokenize', CachedFitTransform(Pipeline([
                ('tokenize', Map(compose([tokenize, normalize_special, unescape]))),
                ('normalize', MapTokens(normalize_elongations)),
            ]), self.memory)),
            ('index', WordIndex(
                # 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
                lambda: get_rng().uniform(-0.25, 0.25, 300),
                joblib.load('../google/GoogleNews-vectors-negative300.pickle') if use_w2v else None,
                include_zero=True
            )),
            ('clip', Clip(56))
        ])
        distant_docs, distant_labels = self.distant_docs[:10**5], self.distant_labels[:10**5]
        feature.fit(distant_docs)
        feature.fit(self.dev_docs)
        feature.fit(self.train_docs)
        classifier = ConvNet(
            batch_size=50, embeddings=feature.named_steps['index'], img_h=56, filter_hs=[3, 4, 5],
            hidden_units=[100, 3], dropout_rates=[0.5], conv_non_linear=rectify, activations=(identity,),
            static_mode=2, lr_decay=0.95, norm_lim=3
        )
        fit_args = dict(
            dev_X=feature.transform(self.dev_docs), dev_y=np.fromiter(self.dev_labels, 'int32'), average_classes=[0, 2]
        )
        classifier.fit(
            feature.transform(distant_docs), np.fromiter(distant_labels, 'int32'),
            shuffle_batch=False, n_epochs=1, **fit_args
        )
        classifier.fit(
            feature.transform(self.train_docs), np.fromiter(self.train_labels, 'int32'),
            shuffle_batch=True, n_epochs=16, **fit_args
        )
        estimator = Pipeline([('features', feature), ('logreg', classifier)])
        return name, estimator

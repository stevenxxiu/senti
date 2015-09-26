
from lasagne.nonlinearities import identity, rectify
from sklearn.externals import joblib
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.svm import SVC

from senti.cache import CachedFitTransform
from senti.features import *
from senti.models import *
from senti.preprocess import *
from senti.rand import get_rng
from senti.transforms import *
from senti.utils import HeadSr

__all__ = ['AllPipelines']


class AllPipelines:
    def __init__(self, unsup_docs, dev_docs, dev_labels, test_docs):
        self.unsup_docs = unsup_docs
        self.dev_docs = dev_docs
        self.test_docs = test_docs
        self.dev_labels = dev_labels
        self.memory = Memory(cachedir='cache', verbose=0)

    def get_svm_pipeline(self):
        tokenize_sense = Map([tokenize, normalize, unescape])
        tokenize_insense = Map([tokenize, str.lower, normalize, unescape])
        tokenize_insense_raw = Map([tokenize, str.lower, unescape])
        features = [
            ('word_n_grams', FeatureUnion([(n, CachedFitTransform(Pipeline([
                ('tokenize', tokenize_insense),
                ('negations', Negations()),
                ('ngrams', WordNGrams(n)),
                ('count', Count()),
                ('binarize', Binarizer(copy=False)),
            ]), self.memory)) for n in range(1, 4 + 1)])),
            ('char_n_grams', FeatureUnion([(n, CachedFitTransform(Pipeline([
                ('tokenize', tokenize_insense),
                ('ngrams', CharNGrams(n)),
                ('proportion', Proportion()),
            ]), self.memory)) for n in range(3, 5 + 1)])),
            ('all_caps', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', AllCaps()),
                ('proportion', Proportion()),
            ])),
            ('punctuations', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Punctuations()),
                ('proportion', Proportion()),
            ])),
            ('elongated', Pipeline([
                ('tokenize', tokenize_insense_raw),
                ('feature', Elongations()),
                ('proportion', Proportion()),
            ])),
            ('emoticons', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Emoticons()),
                ('proportion', Proportion()),
            ])),
            ('emoticon_last', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Emoticons()),
                ('last', Index(-1)),
            ])),
        ]
        name = 'svm({})'.format(','.join(name for name, estimator in features))
        pipeline = Pipeline([
            ('features', FeatureUnion(features)),
            ('svm', SVC(kernel='linear', C=0.005)),
        ])
        return name, pipeline

    def get_logreg_pipeline(self):
        tokenize_sense = Map([tokenize, normalize, unescape])
        features = [
            ('w2v_doc', CachedFitTransform(Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Doc2VecTransform(
                    list(tokenize_sense.transform(docs) for docs in (
                        HeadSr(self.unsup_docs, 10**6), self.dev_docs, self.test_docs
                    )), cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
                )),
            ]), self.memory)),
            # ('w2v_word_avg_google', CachedFitTransform(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', (Word2VecAverage(
            #         word2vec=joblib.load('../google/GoogleNews-vectors-negative300.pickle')
            #     )))
            # ]), self.memory)),
            # ('w2v_word_avg', CachedFitTransform(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecAverage(
            #         tokenize_sense.transform(HeadSr(self.unsup_docs, 10**6)),
            #         cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
            #     )),
            # ]), self.memory)),
            # ('w2v_word_avg_google', CachedFitTransform(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', (Word2VecAverage(
            #         word2vec=joblib.load('../google/GoogleNews-vectors-negative300.pickle')
            #     )))
            # ]), self.memory)),
            # ('w2v_word_max', CachedFitTransform(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecMax(
            #         tokenize_sense.transform(HeadSr(self.unsup_docs, 10**6)),
            #         cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
            #     ))
            # ]), self.memory)),
            # ('w2v_word_inv', CachedFitTransform(Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecInverse(
            #         list(tokenize_sense.transform(docs) for docs in (
            #             HeadSr(self.unsup_docs, 10**5), self.dev_docs, self.test_docs
            #         )), cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
            #     ))
            # ]), self.memory)),
        ]
        name = 'logreg({})'.format(','.join(name for name, estimator in features))
        pipeline = Pipeline([
            ('features', FeatureUnion(features)),
            ('logreg', LogisticRegression()),
        ])
        return name, pipeline

    def get_cnn_pipeline(self):
        use_w2v = True
        name = 'cnn(use_w2v={})'.format(use_w2v)
        input_pipeline = Pipeline([
            ('tokenize', Map([tokenize, normalize])),
            ('index', WordIndex(
                # 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
                lambda: get_rng().uniform(-0.25, 0.25, 300),
                joblib.load('../google/GoogleNews-vectors-negative300.pickle') if use_w2v else None,
                include_zero=True
            )),
            ('clip', Clip(56))
        ])
        input_pipeline.fit(self.dev_docs)
        pipeline = Pipeline(input_pipeline.steps + [
            ('cnn', ConvNet(
                batch_size=50, shuffle_batch=True, n_epochs=6, dev_X=input_pipeline.transform(self.dev_docs),
                dev_y=self.dev_labels, average_classes=[0, 2], embeddings=input_pipeline.named_steps['index'], img_h=56,
                filter_hs=[3, 4, 5], hidden_units=[100, 3], dropout_rates=[0.5], conv_non_linear=rectify,
                activations=(identity,), non_static=True, lr_decay=0.95, norm_lim=3
            )),
        ])
        return name, pipeline

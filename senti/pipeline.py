
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, Normalizer

from senti.features import *
from senti.models import *
from senti.persist import CachedFitTransform
from senti.preprocess import *
from senti.transforms import *

__all__ = ['get_voting_pipeline', 'get_logreg_pipeline', 'get_cnn_pipeline']


def get_bag_features(dev_docs, unsup_docs, unsup_docs_inv):
    memory = Memory(cachedir='cache', verbose=0)
    case_sense = MapTransform([tokenize, normalize])
    case_insense = MapTransform([tokenize, str.lower, normalize])
    return [
        ('all_caps', Pipeline([('tokenizer', case_sense), ('feature', AllCaps())])),
        ('punctuations', Pipeline([('tokenizer', case_sense), ('feature', Punctuations())])),
        ('emoticons', Pipeline([('tokenizer', case_sense), ('feature', Emoticons())])),
        ('word_n_grams', FeatureUnion([(n, Pipeline([
            ('tokenizer', case_insense),
            ('negations', Negations()),
            ('ngrams', WordNGrams(n)),
            # ('binarizer', Binarizer()),
        ])) for n in range(3, 5 + 1)])),
        ('char_n_grams', FeatureUnion([(n, Pipeline([
            ('tokenizer', case_insense),
            ('ngrams', CharNGrams(n)),
            # ('normalizer', Normalizer('l1')),
        ])) for n in range(2, 4 + 1)])),
        ('elongations', Pipeline([('tokenizer', case_insense), ('feature', Elongations())])),
        # ('w2v_doc', Pipeline([('tokenizer', case_insense), ('feature', CachedFitTransform(Doc2VecTransform(
        #     case_insense.transform(dev_docs), case_insense.transform(unsup_docs), cbow=0,
        #     size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
        # ), memory))])),
        ('w2v_word_avg', Pipeline([('tokenizer', case_insense), ('feature', CachedFitTransform(Word2VecAverage(
            case_insense.transform(unsup_docs), cbow=0,
            size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
        ), memory))])),
        # ('w2v_word_avg_google', Pipeline([('tokenizer', case_insense), ('feature', CachedFitTransform(Word2VecAverage(
        #     pretrained_file='../google/GoogleNews-vectors-negative300.bin'
        # ), memory))])),
        # ('w2v_word_max', Pipeline([('tokenizer', case_insense), ('feature', CachedFitTransform(Word2VecMax(
        #     case_insense.transform(unsup_docs), cbow=0,
        #     size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
        # ), memory))])),
        # ('w2v_word_inv', Pipeline([('tokenizer', case_insense), ('feature', CachedFitTransform(Word2VecInverse(
        #     case_insense.transform(dev_docs), case_insense.transform(unsup_docs_inv), cbow=0,
        #     size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
        # ), memory))])),
    ]


def get_voting_pipeline(dev_docs, unsup_docs, unsup_docs_inv):
    features = get_bag_features(dev_docs, unsup_docs, unsup_docs_inv)
    name = 'ensemble_logreg({})'.format(','.join(name for name, estimator in features))
    pipeline = Pipeline([
        ('features', FeatureUnion(features)),
        ('logreg', LogisticRegression()),
    ])
    return name, pipeline


def get_logreg_pipeline(dev_docs, unsup_docs, unsup_docs_inv):
    features = get_bag_features(dev_docs, unsup_docs, unsup_docs_inv)
    name = 'logreg({})'.format(','.join(name for name, estimator in features))
    pipeline = Pipeline([
        ('features', FeatureUnion(features)),
        ('logreg', LogisticRegression()),
    ])
    return name, pipeline


def get_cnn_pipeline(train_docs, dev_docs, dev_y, use_w2v):
    name = 'cnn(use_w2v={})'.format(use_w2v)
    input_pipeline = Pipeline([
        ('case_insense', MapTransform([tokenize, str.lower, normalize])),
        ('index', Index(
            # 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
            lambda num_words: np.random.uniform(-0.25, 0.25, (num_words, 300)),
            Word2Vec().load_binary('../google/GoogleNews-vectors-negative300.bin') if use_w2v else None,
            include_zero=True
        )),
        ('clippad', ClipPad(5 - 1, 56))
    ])
    input_pipeline.fit(train_docs)
    pipeline = Pipeline(input_pipeline.steps + [
        ('cnn', ConvNet(
            input_pipeline.named_steps['index'].X, img_w=300, img_h=64, filter_hs=[3, 4, 5], hidden_units=[100, 3],
            dropout_rates=[0.5], conv_non_linear='relu', activations=(lambda x: x,), non_static=True,
            shuffle_batch=True, n_epochs=8, batch_size=50, lr_decay=0.95, norm_lim=3
        )),
    ])
    return name, pipeline

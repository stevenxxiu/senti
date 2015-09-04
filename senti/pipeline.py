
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, Normalizer

from senti.features import *
from senti.models import VotingClassifier
from senti.persist import CachedFitTransform
from senti.preprocess import *
from senti.transforms import MapTransform

__all__ = ['get_features', 'get_voting_pipeline', 'get_logreg_pipeline']


def get_features(dev_docs, unsup_docs, unsup_docs_inv):
    memory = Memory(cachedir='cache', verbose=0)
    case_sense = MapTransform([tokenize, normalize_urls])
    case_insense = MapTransform([tokenize, str.lower, normalize_urls])
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


def get_voting_pipeline(features):
    return 'ensemble_logreg({})'.format(','.join(name for name, estimator in features)), VotingClassifier(list(
        (name, Pipeline([('feature', estimator), ('logreg', LogisticRegression())]))
        for name, estimator in features
    ), voting='soft')


def get_logreg_pipeline(features):
    return 'logreg({})'.format(','.join(name for name, estimator in features)), Pipeline([
        ('features', FeatureUnion(features)),
        ('logreg', LogisticRegression()),
    ])

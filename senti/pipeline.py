
from senti.features import *
from senti.persist import CachedFitTransform
from senti.preprocess import *
from senti.transforms import MapTransform
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, Normalizer
from sklearn.externals.joblib import Memory
from sklearn.ensemble import VotingClassifier

__all__ = ['get_pipeline_name', 'get_ensemble_pipeline', 'get_logreg_pipeline']


def dfs_pipeline(pipeline, on_start=lambda: None, on_end=lambda desc_num: None):
    if isinstance(pipeline, Pipeline):
        steps = pipeline.steps
    elif isinstance(pipeline, FeatureUnion):
        steps = pipeline.transformer_list
    else:
        raise ValueError
    for i, step in enumerate(steps):
        try:
            yield pipeline, i, step
        except StopIteration:
            continue
        if isinstance(step[1], (Pipeline, FeatureUnion)):
            on_start()
            j = -1
            for j, node_data in enumerate(dfs_pipeline(pipeline, on_start, on_end)):
                yield node_data
            on_end(j + 1)


def get_pipeline_name(pipeline):
    parts = []
    dfs_iter = dfs_pipeline(pipeline, lambda: parts.append('('), lambda: parts.append(')'))
    for parent, i, step in dfs_iter:
        if not isinstance(step[0], str):
            dfs_iter.throw(StopIteration)
        if step[0] != 'features':
            parts.append(step[1])
            parts.append(',')
    return ''.join(parts).replace(',)', ')')


def get_features(dev_docs, unsup_docs):
    memory = Memory(cachedir='cache', verbose=0)
    case_sense_tokenizer = MapTransform([tokenize, normalize_urls])
    case_insense = MapTransform([tokenize, str.lower, normalize_urls])
    return FeatureUnion([
        ('case_sense', Pipeline([
            ('tokenizer', case_sense_tokenizer),
            ('features', FeatureUnion([
                ('all_caps', AllCaps()),
                ('punctuations', Punctuations()),
                ('emoticons', Emoticons()),
            ]))
        ])),
        ('case_insense', Pipeline([
            ('tokenizer', case_insense),
            ('features', FeatureUnion([
                ('word_n_grams', FeatureUnion([(n, Pipeline([
                    # ('negations', Negations()),
                    ('ngrams', WordNGrams(n)),
                    # ('binarizer', Binarizer()),
                ])) for n in range(3, 5 + 1)])),
                ('char_n_grams', FeatureUnion([(n, Pipeline([
                    ('ngrams', CharNGrams(n)),
                    # ('normalizer', Normalizer('l1')),
                ])) for n in range(2, 4 + 1)])),
                ('elongations', Elongations()),
                # ('w2v_doc', CachedFitTransform(Doc2VecTransform(
                #     case_insense.transform(dev_docs), case_insense.transform(unsup_docs), cbow=0,
                #     size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
                # ), memory)),
                # ('w2v_word_avg', CachedFitTransform(Word2VecAverage(
                #     case_insense.transform(unsup_docs), cbow=0,
                #     size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
                # ), memory)),
                # ('w2v_word_avg_google', CachedFitTransform(Word2VecAverage(
                #     case_insense.transform(unsup_docs),
                #     pretrained_file='../google/GoogleNews-vectors-negative300.bin'
                # ), memory)),
                # ('w2v_word_max', CachedFitTransform(Word2VecMax(
                #     case_insense.transform(unsup_docs), cbow=0,
                #     size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
                # ), memory)),
                # ('w2v_word_inv', CachedFitTransform(Word2VecInverse(
                #     case_insense.transform(dev_docs), case_insense.transform(unsup_docs), cbow=0,
                #     size=100, window=10, negative=5, hs=0, sample=1e-4, threads=64, iter=20, min_count=1
                # ), memory)),
            ]))
        ]))
    ])


def get_ensemble_pipeline(dev_docs, unsup_docs):
    features = get_features(dev_docs, unsup_docs)
    replace = []
    dfs_iter = dfs_pipeline(
        features, on_end=lambda desc_num: replace.append((parent, i, step)) if desc_num > 0 else None
    )
    for parent, i, step in dfs_iter:
        if not isinstance(step[0], str):
            dfs_iter.throw(StopIteration)
    for parent, i, step in replace:
        parent[i] = ('features', Pipeline([step, ('logreg', LogisticRegression())]))
    return Pipeline([
        ('features', features),
        ('voting', VotingClassifier()),
    ])


def get_logreg_pipeline(dev_docs, unsup_docs):
    return Pipeline([
        ('features', get_features(dev_docs, unsup_docs)),
        ('logreg', LogisticRegression()),
    ])

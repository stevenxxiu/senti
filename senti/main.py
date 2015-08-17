#!/usr/bin/env python

import json
import os

import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, Normalizer

from senti.features import *
from senti.persist import CachedFitTransform
from senti.preprocess import *
from senti.score import write_score
from senti.transforms import MapTransform
from senti.utils import indexes_of


class PicklableSr:
    def __init__(self, sr):
        self.sr = sr
        self.name = sr.name
        self.encoding = sr.encoding

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['sr']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sr = open(self.name, encoding=self.encoding)


class FieldExtractor(PicklableSr):
    def __init__(self, sr, field):
        super().__init__(sr)
        self.field = field

    def __iter__(self):
        self.sr.seek(0)
        for line in self.sr:
            yield json.loads(line)[self.field]


class HeadSr(PicklableSr):
    def __init__(self, sr, n):
        super().__init__(sr)
        self.n = n

    def __iter__(self):
        self.sr.seek(0)
        for i, line in zip(range(self.n), self.sr):
            yield line


def get_pipeline_name(pipeline):
    if isinstance(pipeline, Pipeline):
        steps = pipeline.steps
    elif isinstance(pipeline, FeatureUnion):
        steps = pipeline.transformer_list
    else:
        raise ValueError
    parts = []
    for step in steps:
        if isinstance(step[0], str):
            part = ''
            if step[0] != 'features':
                part += step[0]
            if isinstance(step[1], (Pipeline, FeatureUnion)):
                name = get_pipeline_name(step[1])
                if name:
                    part += '({})'.format(name)
            parts.append(part)
    return ','.join(parts)


def main():
    # fit & predict
    os.chdir('../data/twitter')
    with open('train.json') as train_sr, open('unsup.txt', encoding='ISO-8859-2') as unsup_sr, \
            open('dev.json') as dev_sr:
        train_docs = FieldExtractor(train_sr, 'text')
        unsup_docs = HeadSr(unsup_sr, 10**6)
        dev_docs = FieldExtractor(dev_sr, 'text')
        memory = Memory(cachedir='cache', verbose=0)
        case_sense_tokenizer = MapTransform([tokenize, normalize_urls])
        case_insense = MapTransform([tokenize, str.lower, normalize_urls])
        pipeline = Pipeline([
            ('features', FeatureUnion([
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
            ])),
            ('logreg', LogisticRegression()),
        ])
        pipeline.fit(train_docs, np.fromiter(FieldExtractor(train_sr, 'label'), int))
        all_probs = pipeline.predict_proba(dev_docs)

    classes = pipeline.steps[-1][1].classes_
    pipeline_name = get_pipeline_name(pipeline)
    os.makedirs('results', exist_ok=True)

    # write predictions
    with open('dev.json') as dev_sr, open('results/{}.json'.format(pipeline_name), 'w') as results_sr:
        for line, probs in zip(dev_sr, all_probs):
            indexes = indexes_of(classes, [0, 1, 2])
            results_sr.write(json.dumps({
                'id': json.loads(line)['id'], 'label': int(classes[np.argmax(probs)]),
                'prob_neg': probs[indexes[0]], 'prob_nt': probs[indexes[1]], 'prob_pos': probs[indexes[2]]
            }) + '\n')

    # write scores
    with open('dev.json') as sr:
        gold_labels = np.fromiter(FieldExtractor(sr, 'label'), int)
    write_score('results/{}'.format(pipeline_name), gold_labels, all_probs, classes, (0, 2))

if __name__ == '__main__':
    main()

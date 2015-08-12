#!/usr/bin/env python

import json
import os

import numpy as np
from functional import compose
from senti.features import *
from senti.preprocess import *
from senti.score import write_score
from senti.utils import indexes_of
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


class FieldExtractor:
    def __init__(self, sr, field):
        self.sr = sr
        self.field = field

    def __iter__(self):
        self.sr.seek(0)
        for line in self.sr:
            yield json.loads(line)[self.field]


def main():
    # fit & predict
    os.chdir('../data/twitter')
    with open('train.json') as train_sr, open('test.json') as unsup_sr, open('dev.json') as dev_sr:
        dev_docs = FieldExtractor(dev_sr, 'text')
        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('all_caps', AllCaps(normalize_urls, tokenize)),
                ('w2v_doc', Word2VecDocs(
                    compose(str.lower, normalize_urls), tokenize, dev_docs, FieldExtractor(unsup_sr, 'text'),
                    cbow=0, size=100, window=10, negative=5, hs=0, sample=1e-4, threads=8, iter=20, min_count=1
                )),
                # ('w2v_word_avg', Word2VecDocs(compose(str.lower, normalize_urls), tokenize)),
                # ('w2v_word_max', Word2VecDocs(compose(str.lower, normalize_urls), tokenize)),
                # ('w2v_word_inv', Word2VecDocs(compose(str.lower, normalize_urls), tokenize)),
            ])),
            ('classifier', LogisticRegression())
        ])
        pipeline.fit(FieldExtractor(train_sr, 'text'), np.fromiter(FieldExtractor(train_sr, 'label'), int))
        all_probs = pipeline.predict_proba(dev_docs)
        classes = pipeline.steps[-1][1].classes_

    # write predictions
    os.makedirs('predictions', exist_ok=True)
    with open('dev.json') as dev_sr, open('predictions/results.json', 'w') as results_sr:
        for line, probs in zip(dev_sr, all_probs):
            indexes = indexes_of(classes, [0, 1, 2])
            results_sr.write(json.dumps({
                'id': json.loads(line)['id'], 'label': int(classes[np.argmax(probs)]),
                'prob_neg': probs[indexes[0]], 'prob_nt': probs[indexes[1]], 'prob_pos': probs[indexes[2]]
            }) + '\n')

    # write scores
    os.makedirs('scores', exist_ok=True)
    with open('dev.json') as sr:
        gold_labels = np.fromiter(FieldExtractor(sr, 'label'), int)
    write_score('scores/results.pdf', gold_labels, all_probs, classes, (0, 2))

if __name__ == '__main__':
    main()

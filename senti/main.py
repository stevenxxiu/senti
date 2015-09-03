#!/usr/bin/env python

import json
import os

import numpy as np
from senti.score import write_score
from senti.utils import indexes_of
from senti.pipeline import *


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


def main():
    # fit & predict
    os.chdir('../data/twitter')
    with open('train.json') as train_sr, open('unsup.txt', encoding='ISO-8859-2') as unsup_sr, \
            open('dev.json') as dev_sr:
        train_docs = FieldExtractor(train_sr, 'text')
        unsup_docs = HeadSr(unsup_sr, 10**6)
        dev_docs = FieldExtractor(dev_sr, 'text')
        pipeline = get_ensemble_pipeline(dev_docs, unsup_docs)
        # pipeline = get_logreg_pipeline()
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

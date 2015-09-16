#!/usr/bin/env python

import json
import os

import numpy as np

from senti.pipeline import *
from senti.rand import set_rng
from senti.score import *
from senti.utils import *


def main():
    # fix seed for reproducibility
    set_rng(np.random.RandomState(1000))

    # fit & predict
    os.chdir('data/twitter')
    with open('train.json') as train_sr, open('unsup.txt', encoding='ISO-8859-2') as unsup_sr, \
            open('dev.json') as dev_sr:
        train_docs = FieldExtractor(train_sr, 'text')
        train_labels = np.fromiter(FieldExtractor(train_sr, 'label'), 'int32')
        unsup_docs = HeadSr(unsup_sr, 10**6)
        unsup_docs_inv = HeadSr(unsup_sr, 10**5)
        dev_docs = FieldExtractor(dev_sr, 'text')
        dev_labels = np.fromiter(FieldExtractor(dev_sr, 'label'), 'int32')
        all_pipelines = AllPipelines(dev_docs, dev_labels, unsup_docs, unsup_docs_inv)
        # pipeline_name, pipeline = all_pipelines.get_logreg_pipeline()
        pipeline_name, pipeline = all_pipelines.get_cnn_pipeline()
        # pipeline_name, pipeline = all_pipelines.get_vote_ensemble_pipeline()
        pipeline.fit(train_docs, train_labels)
        all_probs = pipeline.predict_proba(dev_docs)

    classes = pipeline.classes_
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
    write_score('results/{}'.format(pipeline_name), dev_labels, all_probs, classes, (0, 2))

if __name__ == '__main__':
    main()

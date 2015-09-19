#!/usr/bin/env python

import json
import os

import numpy as np

from senti.pipeline import *
from senti.rand import set_rng
from senti.score import *
from senti.utils import *


def main():
    os.chdir('data/twitter')
    with open('train.json') as train_sr, open('unsup.txt', encoding='ISO-8859-2') as unsup_sr, \
            open('dev.json') as dev_sr, open('test.json') as test_sr:
        train_docs = FieldExtractor(train_sr, 'text')
        train_labels = np.fromiter(FieldExtractor(train_sr, 'label'), 'int32')
        unsup_docs = HeadSr(unsup_sr, 10**6)
        dev_docs = FieldExtractor(dev_sr, 'text')
        dev_labels = np.fromiter(FieldExtractor(dev_sr, 'label'), 'int32')
        test_docs = FieldExtractor(test_sr, 'text')
        test_labels = np.fromiter(FieldExtractor(test_sr, 'label'), 'int32')

        # fix seed for reproducibility
        set_rng(np.random.RandomState(1000))

        # train
        all_pipelines = AllPipelines(unsup_docs, dev_docs, dev_labels, test_docs)
        pipeline_name, pipeline = all_pipelines.get_svm_pipeline()
        # pipeline_name, pipeline = all_pipelines.get_cnn_pipeline()
        # pipeline_name, pipeline = all_pipelines.get_vote_ensemble_pipeline()
        pipeline.fit(train_docs, train_labels)
        classes = pipeline.classes_

        # test_data = [('dev', dev_docs, dev_labels)]
        test_data = [('dev', dev_docs, dev_labels), ('test', test_docs, test_labels)]

        # predict & write results
        for name, docs, labels in test_data:
            os.makedirs('results/{}'.format(name), exist_ok=True)
            all_probs = pipeline.predict_proba(docs)
            with open('{}.json'.format(name)) as sr, \
                    open('results/{}/{}.json'.format(name, pipeline_name), 'w') as results_sr:
                for line, probs in zip(sr, all_probs):
                    indexes = indexes_of(classes, [0, 1, 2])
                    results_sr.write(json.dumps({
                        'id': json.loads(line)['id'], 'label': int(classes[np.argmax(probs)]),
                        'prob_neg': probs[indexes[0]], 'prob_nt': probs[indexes[1]], 'prob_pos': probs[indexes[2]]
                    }) + '\n')
            print('{} data: '.format(name))
            write_score('results/{}/{}'.format(name, pipeline_name), labels, all_probs, classes, (0, 2))

if __name__ == '__main__':
    main()

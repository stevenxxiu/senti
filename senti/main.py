#!/usr/bin/env python

import json
import os

import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from senti.rand import set_rng
from senti.score import *
from senti.sentimodels import *
from senti.utils import *


def main():
    os.chdir('data/twitter')
    with open('train.json') as train_sr, open('unsup.txt', encoding='utf-8') as unsup_sr, \
            open('distant.json') as distant_sr, open('dev.json') as dev_sr, open('test.json') as test_sr:
        train_docs = FieldExtractor(train_sr, 'text')
        train_labels = np.fromiter(FieldExtractor(train_sr, 'label'), 'int32')
        dev_docs = FieldExtractor(dev_sr, 'text')
        dev_labels = FieldExtractor(dev_sr, 'label')
        distant_docs = FieldExtractor(distant_sr, 'text')
        distant_labels = FieldExtractor(distant_sr, 'label')
        test_docs = FieldExtractor(test_sr, 'text')
        test_labels = FieldExtractor(test_sr, 'label')

        # fix seed for reproducibility
        set_rng(np.random.RandomState(1000))

        # train
        senti_models = SentiModels(
            train_docs, train_labels, distant_docs, distant_labels, unsup_sr, dev_docs, dev_labels, test_docs
        )
        pipeline_name, pipeline = senti_models.fit_logreg()
        # pipeline_name, pipeline = senti_models.fit_svm()
        # pipeline_name, pipeline = senti_models.fit_cnn()
        classes = pipeline.classes_

        # test_data = [('dev', dev_docs, dev_labels)]
        test_data = [('dev', dev_docs, dev_labels), ('test', test_docs, test_labels)]

        # predict & write results
        for name, docs, labels in test_data:
            os.makedirs('results/{}'.format(name), exist_ok=True)
            try:
                all_probs = pipeline.predict_proba(docs)
            except AttributeError:
                all_probs = LabelBinarizer().fit(classes).transform(pipeline.predict(docs))
            with open('{}.json'.format(name)) as sr, \
                    open('results/{}/{}.json'.format(name, pipeline_name), 'w') as results_sr:
                for line, probs in zip(sr, all_probs):
                    indexes = LabelEncoder().fit(classes).transform([0, 1, 2])
                    results_sr.write(json.dumps({
                        'id': json.loads(line)['id'], 'label': int(classes[np.argmax(probs)]),
                        'prob_neg': float(probs[indexes[0]]),
                        'prob_nt': float(probs[indexes[1]]),
                        'prob_pos': float(probs[indexes[2]]),
                    }) + '\n')
            print('{} data: '.format(name))
            labels = np.fromiter(labels, dtype='int32')
            write_score('results/{}/{}'.format(name, pipeline_name), labels, all_probs, classes, (0, 2))

if __name__ == '__main__':
    main()

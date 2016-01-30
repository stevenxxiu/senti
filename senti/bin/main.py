#!/usr/bin/env python

import json
import logging
import os
import sys
from contextlib import ExitStack

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from senti.rand import seed_rng
from senti.score import *
from senti.senti_models import *
from senti.utils import BalancedSlice, FieldExtractor, JSONDecoder, RepeatSr, temp_chdir


class SentiData:
    def __init__(self):
        self._stack = ExitStack()
        self.distant_docs = []
        self.distant_labels = []
        self.unsup_docs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        return self._stack.__exit__(*exc_details)


class TwitterData(SentiData):
    def __init__(self):
        super().__init__()
        stack = self._stack
        # classes
        self.classes_ = [0, 1, 2]
        self.average_classes = [0, 2]
        # data
        self.data_dir = 'data/twitter/semeval_2016'
        with temp_chdir(self.data_dir):
            self.train_objs = JSONDecoder(stack.enter_context(open('train.json')))
            self.train_docs = FieldExtractor(self.train_objs, 'text')
            self.train_labels = np.fromiter(FieldExtractor(self.train_objs, 'label'), 'int32')
            distant_srs = [stack.enter_context(open('../emote/class_{}.txt'.format(i), encoding='utf-8')) for i in [0, 2]]
            self.distant_docs = BalancedSlice(distant_srs)
            self.distant_labels = BalancedSlice((RepeatSr(0), RepeatSr(2)))
            unsup_sr = stack.enter_context(open('../unsup/all.txt', encoding='utf-8'))
            self.unsup_docs = BalancedSlice([unsup_sr])
            self.val_objs = JSONDecoder(stack.enter_context(open('val.json')))
            self.val_docs = FieldExtractor(self.val_objs, 'text')
            self.val_labels = FieldExtractor(self.val_objs, 'label')
            self.test_objs = JSONDecoder(stack.enter_context(open('test.json')))
            self.test_docs = FieldExtractor(self.test_objs, 'text')
            self.test_labels = FieldExtractor(self.test_objs, 'label')


class IMDBData(SentiData):
    def __init__(self):
        super().__init__()
        stack = self._stack
        # classes
        self.classes_ = [0, 1, 2]
        self.average_classes = [0, 2]
        # data
        self.data_dir = 'data/imdb'
        with temp_chdir(self.data_dir):
            self.train_objs = JSONDecoder(stack.enter_context(open('train.json')))
            self.train_docs = FieldExtractor(self.train_objs, 'text')
            self.train_labels = np.fromiter(FieldExtractor(self.train_objs, 'label'), 'int32')
            unsup_sr = stack.enter_context(open('unsup.json'))
            self.unsup_docs = BalancedSlice([FieldExtractor(unsup_sr, 'text')])
            self.val_objs = JSONDecoder(stack.enter_context(open('val.json')))
            self.val_docs = FieldExtractor(self.val_objs, 'text')
            self.val_labels = FieldExtractor(self.val_objs, 'label')
            self.test_objs = JSONDecoder(stack.enter_context(open('test.json')))
            self.test_docs = FieldExtractor(self.test_objs, 'text')
            self.test_labels = FieldExtractor(self.test_objs, 'label')


class YelpData(SentiData):
    def __init__(self):
        super().__init__()
        stack = self._stack
        # classes
        self.classes_ = [1, 2, 3, 4, 5]
        self.average_classes = [1, 2, 3, 4, 5]
        # data
        self.data_dir = 'data/yelp'
        with temp_chdir(self.data_dir):
            self.train_objs = JSONDecoder(stack.enter_context(open('train.json')))
            self.train_docs = FieldExtractor(self.train_objs, 'text')
            self.train_labels = np.fromiter(FieldExtractor(self.train_objs, 'stars'), 'int32')
            self.val_objs = JSONDecoder(stack.enter_context(open('val.json')))
            self.val_docs = FieldExtractor(self.val_objs, 'text')
            self.val_labels = FieldExtractor(self.val_objs, 'stars')
            self.test_objs = JSONDecoder(stack.enter_context(open('test.json')))
            self.test_docs = FieldExtractor(self.test_objs, 'text')
            self.test_labels = FieldExtractor(self.test_objs, 'stars')


def main():
    sys.setrecursionlimit(5000)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data = TwitterData()
    with data:
        # load data

        # fix seed for reproducibility
        seed_rng(1234)

        # train
        senti_models = SentiModels(data)
        # pipeline_name, pipeline = senti_models.fit_voting()
        # pipeline_name, pipeline = senti_models.fit_logreg()
        # pipeline_name, pipeline = senti_models.fit_word2vec_bayes()
        # pipeline_name, pipeline = senti_models.fit_svm()
        pipeline_name, pipeline = senti_models.fit_nn_word()
        # pipeline_name, pipeline = senti_models.fit_cnn_char()
        # pipeline_name, pipeline = senti_models.fit_cnn_word_char()
        # pipeline_name, pipeline = senti_models.fit_rnn_char_cnn_word()

        # test_data = [('val', data.val_objs, data.val_docs, data.val_labels)]
        test_data = [
            ('val', data.val_objs, data.val_docs, data.val_labels),
            ('test', data.test_objs, data.test_docs, data.test_labels)
        ]

        # predict & write results
        for name, objs, docs, labels in test_data:
            try:
                probs = pipeline.predict_proba(docs)
            except AttributeError:
                probs = LabelBinarizer().fit(data.classes_).transform(pipeline.predict(docs))
            results_dir = os.path.join(data.data_dir, 'results', name)
            os.makedirs(results_dir, exist_ok=True)
            with temp_chdir(results_dir):
                with open('{}.json'.format(pipeline_name), 'w') as results_sr:
                    for obj, prob in zip(objs, probs):
                        results_sr.write(json.dumps({
                            'id': obj['id'], 'label': data.classes_[np.argmax(prob)],
                            'probs': [(c, prob.item()) for c, prob in zip(data.classes_, prob)]
                        }) + '\n')
                print('{} data: '.format(name))
                labels = np.fromiter(labels, dtype='int32')
                write_score('{}'.format(pipeline_name), labels, probs, data.classes_, data.average_classes)

if __name__ == '__main__':
    main()

#!/usr/bin/env python

import json
import os

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import precision_recall_fscore_support

from senti.score import *
from senti.utils import *


# noinspection PyTypeChecker
def neg_f1_weighted(w, classifier_probs, dev_labels):
    all_probs = w*classifier_probs[0] + (1 - w)*classifier_probs[1]
    pred_labels = np.argmax(all_probs, axis=1)
    precision, recall, fscore, _ = precision_recall_fscore_support(dev_labels, pred_labels)
    return -np.mean(fscore[np.array([0, 2])])


def main():
    # XXX ugly hack, rewrite later without reading from json
    os.chdir('data/twitter')
    names = [
        'cnn(use_w2v=True)',
        'logreg(all_caps,punctuations,emoticons,word_n_grams,char_n_grams,elongations,w2v_word_avg)'
    ]
    classifier_probs = []
    for name in names:
        all_probs = []
        with open('results/{}.json'.format(name)) as sr:
            for line in sr:
                obj = json.loads(line)
                all_probs.append(np.array([obj['prob_neg'], obj['prob_nt'], obj['prob_pos']]))
        classifier_probs.append(np.vstack(all_probs))
    classifier_probs = np.array(classifier_probs)

    with open('dev.json') as dev_sr:
        dev_labels = np.fromiter(FieldExtractor(dev_sr, 'label'), 'int32')

    w = minimize_scalar(neg_f1_weighted, bounds=(0.75, 1), method='bounded', args=(classifier_probs, dev_labels)).x
    print('w: {:.4f}'.format(w))
    all_probs = w*classifier_probs[0] + (1 - w)*classifier_probs[1]
    pipeline_name = 'vote({})'.format(','.join(names))
    classes = np.array([0, 1, 2])

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

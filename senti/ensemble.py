#!/usr/bin/env python

import json
import os

import numpy as np

from senti.score import *
from senti.utils import *


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

    all_probs = np.mean(classifier_probs, axis=0)
    classes = np.array([0, 1, 2])
    pipeline_name = 'vote({})'.format(','.join(names))

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

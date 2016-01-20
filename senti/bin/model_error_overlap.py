#!/usr/bin/env python

import re
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(confusion, names):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(confusion), cmap=plt.cm.YlOrRd, interpolation='nearest')
    width, height = confusion.shape
    fig.colorbar(res)
    plt.xticks(range(width), names[:width], rotation='vertical')
    plt.yticks(range(height), names[:height])


def main():
    os.chdir('data/twitter')
    names = [
        'cnn_char',
        'cnn_word(embedding=google)',
        # 'cnn_word(embedding=none)',
        # 'cnn_word(embedding=twitter)',
        'cnn_word_pred_interaction(embedding=google)',
        # 'logreg(w2v_doc)',
        # 'logreg(w2v_word_avg)',
        # 'logreg(w2v_word_avg_google)',
        # 'logreg(w2v_word_inv)',
        # 'logreg(w2v_word_max)',
        # 'logreg(w2v_word_max_google)',
        # 'logreg(w2v_word_norm_avg)',
        # 'logreg(w2v_word_norm_avg_google)',
        'multiview_cnn_word_cnn_char(embedding=google)',
        # 'multiview_cnn_word_cnn_char(embedding=none)',
        'rnn_char_cnn_word',
        'rnn_multi_word(embedding=google)',
        'rnn_word(embedding=google)',
        # 'rnn_word(embedding=none)',
        # 'svm(word_n_grams,char_n_grams,all_caps,hashtags,punctuations,punctuation_last,emoticons,emoticon_last,'
        # 'elongated,negation_count)',
        'word2vec_bayes',
    ]
    errors = OrderedDict()
    for name in names:
        errors[name] = set()
        with open('results/test/{}.json'.format(name)) as res_sr, open('semeval/test.json', 'r') as gold_sr:
            for res_line, gold_line in zip(res_sr, gold_sr):
                res_obj, gold_obj = json.loads(res_line), json.loads(gold_line)
                if res_obj['label'] != gold_obj['label']:
                    errors[name].add(res_obj['id'])
    confusion = np.empty([len(names), len(names)])
    for i_1, name_1 in enumerate(names):
        for i_2, name_2 in enumerate(names):
            confusion[i_1, i_2] = len(errors[name_1] & errors[name_2]) / len(errors[name_1] | errors[name_2])
    ax_names = [re.sub(r'\([^()]*\)', '', name) for name in names]
    plot_confusion(confusion, ax_names)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

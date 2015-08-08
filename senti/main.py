#!/usr/bin/env python

import json
import os

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from senti.features import *
from senti.score import write_score
from senti.stream import MergedStream, SourceStream, split_streams
from senti.transforms import *
from senti.utils import indexes_of


def feature_label_vecs(sr):
    # assume sparsity for safety
    data, row, col = [], [], []
    labels = []
    for i, obj in enumerate(sr):
        vec = obj['vec']
        if not isinstance(vec, sparse.coo_matrix):
            vec = sparse.coo_matrix(vec)
        data.extend(vec.data)
        col.extend(vec.col)
        row.extend([i]*len(vec.data))
        labels.append(obj['label'])
    return sparse.coo_matrix((data, (row, col))), labels


def main():
    os.chdir('../data/twitter')

    # features
    normed_sr = NormalizeTransform(MergedStream(list(map(SourceStream, ['train.json', 'dev.json', 'test.json']))))
    all_caps_sr = AllCaps(normed_sr)
    w2v_doc_sr = Word2VecDocs(LowerTransform(normed_sr), reuse=True)
    w2v_word_avg_sr = Word2VecWordAverage(LowerTransform(normed_sr), reuse=True)
    w2v_word_max_sr = Word2VecWordMax(LowerTransform(normed_sr), reuse=True)
    w2v_word_inv_sr = Word2VecInverse(LowerTransform(normed_sr), reuse=True)
    feature_sr = VecConcatTransform(w2v_doc_sr, all_caps_sr)
    feature_srs = split_streams(feature_sr, list(map(SourceStream, ['train.json', 'dev.json', 'test.json'])))

    # train
    vecs, labels = feature_label_vecs(next(feature_srs))
    model = LogisticRegression()
    model.fit(vecs, labels)

    # predict
    vecs, gold_labels = feature_label_vecs(next(feature_srs))
    all_probs = model.predict_proba(vecs)
    os.makedirs('predictions', exist_ok=True)
    with open('predictions/{}.json'.format(feature_sr.name), 'w') as sr:
        for src_obj, probs in zip(SourceStream('dev.json'), all_probs):
            indexes = indexes_of(model.classes_, [0, 1, 2])
            sr.write(json.dumps({
                'id': src_obj['id'], 'label': int(model.classes_[np.argmax(probs)]),
                'prob_neg': probs[indexes[0]], 'prob_nt': probs[indexes[1]], 'prob_pos': probs[indexes[2]]
            }) + '\n')

    # write scores
    os.makedirs('scores', exist_ok=True)
    write_score('scores/{}.pdf'.format(feature_sr.name), gold_labels, np.array(all_probs), model.classes_, (0, 2))

if __name__ == '__main__':
    main()

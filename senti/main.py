#!/usr/bin/env python

import json
import os

import numpy as np

from senti.features import *
from senti.models.liblinear import LibLinear
from senti.score import write_score
from senti.stream import MergedStream, SourceStream
from senti.transforms import *


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

    # train
    vecs = (obj['vec'] for obj in feature_sr)
    model = LibLinear('-s 0')
    model.fit(vecs, (obj['label'] for obj in SourceStream('train.json')))

    # write predictions
    gold_labels = []
    all_probs = []
    os.makedirs('predictions', exist_ok=True)
    with open('predictions/{}.json'.format(feature_sr.name), 'w') as sr:
        for src_obj, probs in zip(SourceStream('dev.json'), model.predict_proba(vecs)):
            sr.write(json.dumps({
                'id': src_obj['id'], 'label': model.classes_[np.argmax(probs)],
                'prob_neg': probs[model.classes_.index(0)],
                'prob_nt': probs[model.classes_.index(1)],
                'prob_pos': probs[model.classes_.index(2)]
            }) + '\n')
            gold_labels.append(src_obj['label'])
            all_probs.append(probs)

    # write scores
    os.makedirs('scores', exist_ok=True)
    write_score('scores/{}.pdf'.format(feature_sr.name), gold_labels, np.array(all_probs), model.classes_, (0, 2))

if __name__ == '__main__':
    main()

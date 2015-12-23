#!/usr/bin/env python

import os
from collections import OrderedDict

import joblib

from senti.utils.gensim_ import *


def main():
    os.chdir('data/google')
    model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=False)
    for v in model.vocab.values():
        v.sample_int = 0
    ts = list(model.vocab.items())
    ts.sort(key=lambda t: t[1].index)
    model.vocab = OrderedDict(ts)
    joblib.dump(model, 'GoogleNews-vectors-negative300.pickle')

if __name__ == '__main__':
    main()

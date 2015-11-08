#!/usr/bin/env python

import os

import joblib

from senti.gensim_ext import *


def main():
    os.chdir('data/google')
    model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=False)
    for v in model.vocab.values():
        v.sample_int = 0
    joblib.dump(model, 'GoogleNews-vectors-negative300.pickle')

if __name__ == '__main__':
    main()

#!/usr/bin/env python

import os

import joblib
import numpy as np

from senti.features.word2vec import Word2Vec


def main():
    os.chdir('data/google')
    word2vec = Word2Vec()
    with open('GoogleNews-vectors-negative300.bin', 'rb') as sr:
        header = sr.readline()
        vocab_size, layer1_size = tuple(map(int, header.split()))
        binary_len = np.float32().itemsize*layer1_size
        vecs = []
        for i in range(vocab_size):
            word = bytearray()
            while True:
                ch = sr.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':
                    # ignore newlines in front of words (some binary files have them)
                    word.append(ch[0])
            word2vec.word_to_index[word.decode('utf-8')] = i
            vecs.append(np.frombuffer(sr.read(binary_len), dtype='float32'))
        word2vec.X = np.vstack(vecs)
    joblib.dump(word2vec, 'GoogleNews-vectors-negative300.pickle')

if __name__ == '__main__':
    main()

#!/usr/bin/env python

from senti.features.word2vec import *
from senti.preprocess import NormTextStream
from senti.stream import MergedStream, SourceStream


def main():
    os.chdir('data/twitter')
    data_files = ['train.json', 'dev.json', 'test.json']

    # throw all files into word2vec
    merged_sr = NormTextStream(MergedStream(list(map(SourceStream, data_files))))
    w2v_doc_sr = Word2VecDocs(merged_sr, reuse=True)
    w2v_word_avg_sr = Word2VecWordMax(merged_sr, reuse=True)
    w2v_word_max_sr = Word2VecWordAverage(merged_sr, reuse=True)
    w2v_word_inv_sr = Word2VecInverse(merged_sr, reuse=True)

    for line in zip(range(100), w2v_word_inv_sr):
        print(line)

if __name__ == '__main__':
    main()

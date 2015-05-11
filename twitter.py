#!/usr/bin/env python

import itertools
import json
import os
import re

data_files = ['data/dev.json', 'data/train.json']
liblinear_files = ['data/dev.txt', 'data/train.txt']


def normalize_text(text):
    return re.sub(r'([\.",()!?;:])', r' \1 ', text.lower())


def word2vec():
    # throw training & test data into word2vec and output the sentence vectors
    with open('data/alldata.txt', 'w') as out_sr:
        for i, line in enumerate(itertools.chain(*(open(name) for name in data_files))):
            text = normalize_text(next(iter(json.loads(line).keys())))
            out_sr.write('_*{} {}\n'.format(i, text))
    os.system(r'''
        time ./word2vec/word2vec -train data/alldata.txt -output data/vectors.txt -cbow 0 -size 100 -window 10
        -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
    '''.replace('\n', ''))
    with open('data/vectors.txt', encoding='ISO-8859-1') as in_sr, open('data/sentence_vectors.txt', 'w') as out_sr:
        for line in in_sr:
            if line.startswith('_*'):
                out_sr.write(line)


def prepare_liblinear():
    with open('data/sentence_vectors.txt') as vect_sr:
        for data_name, liblinear_name in zip(data_files, liblinear_files):
            with open(data_name) as data_sr, open(liblinear_name, 'w') as out_sr:
                for vect_line, data_line in zip(vect_sr, data_sr):
                    classif = next(iter(json.loads(data_line).values()))//5 - 1
                    out_data = ' '.join('{}:{}'.format(i + 1, v) for i, v in enumerate(vect_line.strip().split()[1:]))
                    out_sr.write('{} {}\n'.format(classif, out_data))


def train_predict():
    os.system('liblinear/train -s 0 data/train.txt data/model.logreg')
    os.system('liblinear/predict -b 1 data/dev.txt data/model.logreg data/out.logreg')


def main():
    word2vec()
    prepare_liblinear()
    train_predict()


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import itertools
import json
import os
import re
import numpy as np

data_dir = 'data/pitchwise'
data_files = [os.path.join(data_dir, path) for path in ('train.json', 'dev.json')]
liblinear_files = [os.path.join(data_dir, path) for path in ('train.txt', 'dev.txt')]


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'([\.\",()!?;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def word2vec():
    # throw training & test data into word2vec and output the sentence vectors
    with open('{}/alldata.txt'.format(data_dir), 'w') as out_sr:
        for i, line in enumerate(itertools.chain(*(open(name) for name in data_files))):
            text = normalize_text(next(iter(json.loads(line).keys())))
            out_sr.write('_*{} {}\n'.format(i, text))
    os.system(r'''
        time word2vec/word2vec -train {0}/alldata.txt -output {0}/vectors.txt -cbow 0 -size 100 -window 10
        -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
    '''.replace('\n', '').format(data_dir))
    with open('{}/vectors.txt'.format(data_dir), encoding='ISO-8859-1') as in_sr, \
            open('{}/sentence_vectors.txt'.format(data_dir), 'w') as out_sr:
        for line in in_sr:
            if line.startswith('_*'):
                out_sr.write(line)


def prepare_liblinear():
    with open('{}/sentence_vectors.txt'.format(data_dir)) as vect_sr:
        for data_name, liblinear_name in zip(data_files, liblinear_files):
            with open(data_name) as data_sr, open(liblinear_name, 'w') as out_sr:
                # data_sr first to not over-consume data_sr
                for data_line, vect_line in zip(data_sr, vect_sr):
                    classif = next(iter(json.loads(data_line).values()))
                    if classif is None:
                        continue
                    classif //= 5
                    out_data = ' '.join('{}:{}'.format(i + 1, v) for i, v in enumerate(vect_line.strip().split()[1:]))
                    out_sr.write('{} {}\n'.format(classif, out_data))


def train_dev():
    os.system('liblinear/train -s 0 {0}/train.txt {0}/train.logreg'.format(data_dir))
    os.system('liblinear/predict -b 1 {0}/dev.txt {0}/train.logreg {0}/dev.logreg'.format(data_dir))


def test():
    if not os.path.exists('{0}/test.json'.format(data_dir)):
        return
    # load word2vec vectors
    vects = {}
    with open('{}/vectors.txt'.format(data_dir), encoding='ISO-8859-1') as in_sr:
        for line in in_sr:
            if not line.startswith('_*'):
                tokens = line.strip().split()
                vects[tokens[0]] = np.array(list(map(float, tokens[1:])))
    # convert test data to vectors, skipping unknown words
    with open('{0}/test.json'.format(data_dir), 'r') as in_sr, open('{0}/test.txt'.format(data_dir), 'w') as out_sr:
        for line in in_sr:
            sentence = normalize_text(next(iter(json.loads(line).keys())))
            sentence = normalize_text("Gas by my house hit $3.39!!!! I'm going to Chapel Hill on Sat. :)")
            sentence_vect = np.zeros(len(next(iter(vects.values()))))
            for token in sentence.split():
                if token in vects:
                    sentence_vect += vects[token]
            out_sr.write('-1 {}\n'.format(' '.join('{}:{:.6f}'.format(i + 1, v) for i, v in enumerate(sentence_vect))))


def main():
    word2vec()
    prepare_liblinear()
    train_dev()

if __name__ == '__main__':
    main()

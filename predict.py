#!/usr/bin/env python

import csv
import itertools
import json
import os
import re

data_dir = 'data/pitchwise'
data_files = [os.path.join(data_dir, path) for path in ('train.json', 'dev.json', 'test.json')]
liblinear_files = [os.path.join(data_dir, path) for path in ('train.txt', 'dev.txt', 'test.txt')]


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'([\.\",()!?;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def word2vec():
    # throw training & test data into word2vec and output the sentence vectors
    with open('{}/alldata.txt'.format(data_dir), 'w') as out_sr:
        for i, line in enumerate(itertools.chain(*(open(name) for name in data_files))):
            obj = json.loads(line)
            text = normalize_text(next(iter(obj.keys()))) if len(obj) == 1 else obj['text']
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
                    classif = classif//5 if classif is not None else -1
                    if classif == -1 and os.path.split(data_name)[1] == 'train.json':
                        # logistic regression doesn't use unsupervised training data
                        continue
                    out_data = ' '.join('{}:{}'.format(i + 1, v) for i, v in enumerate(vect_line.strip().split()[1:]))
                    out_sr.write('{} {}\n'.format(classif, out_data))


def train_dev():
    os.system('liblinear/train -s 0 {0}/train.txt {0}/train.logreg'.format(data_dir))
    os.system('liblinear/predict -b 1 {0}/dev.txt {0}/train.logreg {0}/dev.logreg'.format(data_dir))


def test():
    os.system('liblinear/predict -b 1 {0}/test.txt {0}/train.logreg {0}/test.logreg > /dev/null'.format(data_dir))
    # output csv file
    with open('{}/test.json'.format(data_dir)) as json_sr, \
            open('{}/test.logreg'.format(data_dir)) as logreg_sr, \
            open('{}/test.csv'.format(data_dir), 'w') as out_sr:
        out_csv = csv.writer(out_sr)
        out_csv.writerow(['docid', 'sentid', 'class', 'prob_pos', 'prob_nt', 'prob_neg', 'sentence'])
        for json_line, logreg_line in zip(json_sr, itertools.islice(logreg_sr, 1, None)):
            obj = json.loads(json_line)
            out_csv.writerow([obj['docid'], obj['sentid']] + logreg_line.split() + [obj['text']])


def main():
    word2vec()
    prepare_liblinear()
    train_dev()
    test()

if __name__ == '__main__':
    main()

#!/usr/bin/env python

import csv
import itertools
import json
import os
import re
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

data_dir = 'data/twitter'
data_files = [os.path.join(data_dir, path) for path in ('train.json', 'dev.json', 'test.json')]
liblinear_files = [os.path.join(data_dir, path) for path in ('train.txt', 'dev.txt', 'test.txt')]


@contextmanager
def temp_chdir(path):
    prev_path = os.getcwd()
    os.chdir(path)
    yield prev_path
    os.chdir(prev_path)


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'([\.\",()!?;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def iterate_text(sr):
    for line in sr:
        doc = json.loads(line)
        doc_id = doc['id']
        if 'sentences' in doc:
            for sent in doc['sentences']:
                yield ((doc_id, sent['id']), (doc, sent))
        else:
            yield ((doc_id,), (doc,))


def iterate_norm_text(sr):
    for id_, path in iterate_text(sr):
        obj = path[-1]
        if not obj.get('norm', False):
            obj['text'] = normalize_text(obj['text'])
        yield (id_, path)


def word2vec():
    # throw training & test data into word2vec, performing a word embedding
    with open('{}/alldata.txt'.format(data_dir), 'w') as out_sr:
        for data_name in data_files:
            with open(data_name, 'r') as in_sr:
                for id_, path in iterate_norm_text(in_sr):
                    out_sr.write('_*{} {}\n'.format('_'.join(id_), path[-1]['text']))
    os.system(r'''
        time word2vec/word2vec -train {0}/alldata.txt -output {0}/vectors.txt -cbow 0 -size 100 -window 10
        -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
    '''.replace('\n', '').format(data_dir))


def word2vec_words():
    word_to_vecs = {}
    with open('{}/vectors.txt'.format(data_dir), encoding='ISO-8859-1') as in_sr:
        lines = iter(in_sr)
        dims = tuple(map(int, re.match(r'(\d+) (\d+)', next(lines)).groups()))
        for line in lines:
            word, vec = re.match(r'(\S+) (.+)', line).groups()
            word_to_vecs[word] = np.array(list(map(float, vec.strip().split())))
    return dims, word_to_vecs


def word2vec_write_docs():
    with open('{}/vectors.txt'.format(data_dir), encoding='ISO-8859-1') as in_sr, \
            open('{}/sentence_vectors.txt'.format(data_dir), 'w') as out_sr:
        for line in in_sr:
            if line.startswith('_*'):
                out_sr.write(line)


def word2vec_write_average():
    dims, word_to_vecs = word2vec_words()
    with open('{}/sentence_vectors.txt'.format(data_dir), 'w') as out_sr:
        for data_name in data_files:
            with open(data_name, 'r') as in_sr:
                for id_, path in iterate_norm_text(in_sr):
                    words = path[-1]['text'].split()
                    vec = np.vstack(word_to_vecs[word] for word in words if word in word_to_vecs).mean(0)
                    out_sr.write('_*{} {}\n'.format('_'.join(id_), ' '.join(map(str, vec))))


def word2vec_write_max():
    # component-wise abs max, doesn't make much sense because the components aren't importance, but worth a try
    dims, word_to_vecs = word2vec_words()
    with open('{}/sentence_vectors.txt'.format(data_dir), 'w') as out_sr:
        for data_name in data_files:
            with open(data_name, 'r') as in_sr:
                for id_, path in iterate_norm_text(in_sr):
                    words = path[-1]['text'].split()
                    words_matrix = np.vstack(word_to_vecs[word] for word in words if word in word_to_vecs)
                    arg_maxes = abs(words_matrix).argmax(0)
                    vec = words_matrix[arg_maxes, np.arange(len(arg_maxes))]
                    out_sr.write('_*{} {}\n'.format('_'.join(id_), ' '.join(map(str, vec))))


def word2vec_inverse():
    # throw training & test data into word2vec, performing a document embedding
    # number documents using integers for easy sorting later
    word_to_docs = defaultdict(set)
    i = 0
    for data_name in data_files:
        with open(data_name, 'r') as in_sr:
            for id_, path in iterate_norm_text(in_sr):
                words = path[-1]['text'].split()
                for word in words:
                    word_to_docs[word].add(str(i))
                i += 1
    with open('{}/alldata.txt'.format(data_dir), 'w') as out_sr:
        for word, docs in word_to_docs.items():
            # remove common & rare words
            if 10 <= len(docs) <= 2000:
                out_sr.write('_*{} {}\n'.format(word, ' '.join(docs)))
    os.system(r'''
        time word2vec/word2vec -train {0}/alldata.txt -output {0}/vectors.txt -cbow 0 -size 100 -window 10
        -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
    '''.replace('\n', '').format(data_dir))
    with open('{}/vectors.txt'.format(data_dir), encoding='ISO-8859-1') as in_sr, \
            open('{}/sentence_vectors.txt'.format(data_dir), 'w') as out_sr:
        for line in itertools.islice(in_sr, 1, None):
            if not line.startswith('_*') and not line.startswith('</s>'):
                out_sr.write(line)
    os.system('sort -n {0}/sentence_vectors.txt > {0}/sentence_vectors_sorted.txt'.format(data_dir))
    os.system('mv {0}/sentence_vectors_sorted.txt {0}/sentence_vectors.txt'.format(data_dir))


def prepare_liblinear():
    with open('{}/sentence_vectors.txt'.format(data_dir)) as vect_sr:
        for data_name, liblinear_name in zip(data_files, liblinear_files):
            with open(data_name) as in_sr, open(liblinear_name, 'w') as out_sr:
                # text first to not over-consume vect_sr
                for (id_, path), vect_line in zip(iterate_text(in_sr), vect_sr):
                    label = path[-1].get('label', -1)
                    out_data = ' '.join('{}:{}'.format(i + 1, v) for i, v in enumerate(vect_line.strip().split()[1:]))
                    out_sr.write('{} {}\n'.format(label, out_data))


def train():
    os.system('liblinear/train -s 0 {0}/train.txt {0}/train.logreg'.format(data_dir))


def test(test_name):
    os.system('liblinear/predict -b 1 {0}/{1}.txt {0}/train.logreg {0}/{1}.logreg'.format(data_dir, test_name))
    with open('{0}/{1}.json'.format(data_dir, test_name)) as in_sr, \
            open('{0}/{1}.logreg'.format(data_dir, test_name)) as logreg_sr, \
            open('{0}/{1}_out.json'.format(data_dir, test_name), 'w') as out_sr:
        prev_path = (None,)
        for (id_, path), logreg_line in zip(iterate_text(in_sr), itertools.islice(logreg_sr, 1, None)):
            if prev_path[0] and prev_path[0] != path[0]:
                out_sr.write(json.dumps(prev_path[0]) + '\n')
            label, prob_pos, prob_neg, prob_nt = logreg_line.split()
            path[-1].update({'label': int(label), 'prob_pos': prob_pos, 'prob_neg': prob_neg, 'prob_nt': prob_nt})
            prev_path = path
        if prev_path[0]:
            out_sr.write(json.dumps(prev_path[0]) + '\n')


def test_dev():
    test('dev')


def test_semeval():
    test('test')
    with open('{}/test_out.json'.format(data_dir)) as in_sr, open('{}/test_out.txt'.format(data_dir), 'w') as out_sr:
        for id_, path in iterate_text(in_sr):
            out_sr.write('NA\t{}\t{}\n'.format(id_[0], ['negative', 'neutral', 'positive'][path[-1]['label']]))
    with temp_chdir('scoring'):
        os.system('./score-semeval2015-task10-subtaskB.pl ../{}/test_out.txt'.format(data_dir))


def test_pitchwise():
    test('test')
    with open('{}/test_out.json'.format(data_dir)) as in_sr, open('{}/test.csv'.format(data_dir), 'w') as out_sr:
        out_csv = csv.writer(out_sr)
        out_csv.writerow(['docid', 'sentid', 'class', 'prob_pos', 'prob_neg', 'prob_nt', 'sentence'])
        for id_, path in iterate_text(in_sr):
            obj = path[-1]
            out_csv.writerow(id_ + [obj['label'], obj['prob_pos'], obj['prob_neg'], obj['prob_nt'], obj['text']])


def main():
    # word2vec()
    # word2vec_write_docs()
    # word2vec_write_average()
    # word2vec_write_max()
    word2vec_inverse()
    prepare_liblinear()
    train()
    test_dev()
    test_semeval()

if __name__ == '__main__':
    main()

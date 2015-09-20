#!/usr/bin/env python

import json
import os
import re


def unique(seq):
    seen = set()
    seen_add = seen.add
    return list(x for x in seq if not (x in seen or seen_add(x)))


def main():
    os.chdir('data/twitter')
    class_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    for name, path in [
        ('dev', 'input/unitn/dev/gold/twitter-dev-gold-B.tsv'),
        ('train', 'input/unitn/train/cleansed/twitter-train-cleansed-B.txt')
    ]:
        with open(path) as sr, open('{}.json'.format(name), 'w') as out_sr:
            for line in sr:
                doc_id, label, text = re.match(r'\d+\t(\d+)\t(negative|neutral|positive)\t(.+)', line).groups()
                text = text.encode().decode('unicode-escape')
                out_sr.write(json.dumps({'id': doc_id, 'text': text, 'label': class_map[label]}) + '\n')
    with open('input/test/SemEval2015-task10-test-B-input.txt') as in_sr, \
            open('input/test/SemEval2015-task10-test-B-gold.txt') as labels_sr, open('test.json', 'w') as out_sr:
        for line, label_line in zip(in_sr, labels_sr):
            doc_id, text = re.match(r'NA\t(T\d+)\tunknwn\t(.+)', line).groups()
            doc_id_label, label = re.match(r'\d+\t(T\d+)\t(negative|neutral|positive)', label_line).groups()
            assert doc_id == doc_id_label
            out_sr.write(json.dumps({'id': doc_id, 'text': text, 'label': class_map[label]}) + '\n')

if __name__ == '__main__':
    main()

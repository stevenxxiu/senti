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
    for name in ('dev', 'train'):
        with open('input/duplicated/{}.json'.format(name)) as in_sr, open('{}.json'.format(name), 'w') as out_sr:
            lines = unique(in_sr.read().splitlines())
            for i, line in enumerate(lines):
                text, label = next(iter(json.loads(line).items()))
                out_sr.write(json.dumps({'id': '{}_{}'.format(name, i), 'text': text, 'label': int(label)//5}) + '\n')
    with open('input/test/SemEval2015-task10-test-B-input.txt') as in_sr, \
            open('input/test/SemEval2015-task10-test-B-gold.txt') as labels_sr, open('test.json', 'w') as out_sr:
        for line, label_line in zip(in_sr, labels_sr):
            doc_id, text = re.match(r'^NA\t(T\d+)\tunknwn\t(.+)', line).groups()
            doc_id_label, label = re.match(r'\d+\t(T\d+)\t(negative|neutral|positive)', label_line).groups()
            assert doc_id == doc_id_label
            out_sr.write(json.dumps({
                'id': doc_id, 'text': text, 'label': {'negative': 0, 'neutral': 1, 'positive': 2}[label]
            }) + '\n')

if __name__ == '__main__':
    main()

#!/usr/bin/env python

import json
import re
import os


def main():
    os.chdir('data/twitter')
    for name in ('dev', 'train'):
        with open('input/{}.json'.format(name)) as in_sr, open('{}.json'.format(name), 'w') as out_sr:
            for i, line in enumerate(in_sr):
                text, label = next(iter(json.loads(line).items()))
                out_sr.write(json.dumps({'doc_id': str(i), 'text': text, 'label': int(label)//5}) + '\n')
    with open('input/SemEval2015-task10-test-B-input.txt') as in_sr, open('test.json', 'w') as out_sr:
        for line in in_sr:
            doc_id, text = re.match(r'^NA\t(T[0-9]+)\tunknwn\t(.+)', line).groups()
            out_sr.write(json.dumps({'doc_id': doc_id, 'text': text}) + '\n')

if __name__ == '__main__':
    main()

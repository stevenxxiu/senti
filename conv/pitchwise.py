#!/usr/bin/env python

import json
import os
import re


def main():
    os.chdir('data/pitchwise')
    with open('input/test.txt') as in_sr, open('test.json', 'w') as out_sr:
        for line in in_sr:
            doc_id, line = re.match(r'(\d+) (.+)$', line).groups()
            if not line:
                continue
            sents = []
            for i, sent in enumerate(line.split('|||')):
                if not sent:
                    continue
                sents.append({'id': str(i), 'text': sent, 'norm': True})
            out_sr.write(json.dumps({'id': doc_id, 'sentences': sents}) + '\n')

if __name__ == '__main__':
    main()

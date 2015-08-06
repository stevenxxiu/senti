#!/usr/bin/env python

import json
import os
import re


def main():
    os.chdir('data/pitchwise')
    with open('input/test.txt') as in_sr, open('test.json', 'w') as out_sr:
        for line in in_sr:
            doc_id, line = re.match(r'(\d+) (.+)$', line).groups()
            for i, sent in enumerate(line.split('|||')):
                if not sent:
                    continue
                out_sr.write(json.dumps({'id': '{}_{}'.format(doc_id, id), 'text': sent}) + '\n')

if __name__ == '__main__':
    main()

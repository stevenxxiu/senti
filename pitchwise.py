#!/usr/bin/env python

import json
import re


def to_json(in_path, out_path):
    with open(in_path, 'r') as in_sr, open(out_path, 'w') as out_sr:
        for line in in_sr:
            doc_id, line = re.match(r'(\d+) (.+)$', line).groups()
            doc_id = int(doc_id)
            for i, sent in enumerate(line.split('|||')):
                if not sent:
                    continue
                json.dump({'docid': doc_id, 'sentid': i, 'text': sent.strip()}, out_sr)
                out_sr.write('\n')


def main():
    to_json('data/pitchwise/test.dat', 'data/pitchwise/test.json')

if __name__ == '__main__':
    main()

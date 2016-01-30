#!/usr/bin/env python

import json
import os


def main():
    os.chdir('data/twitter/semeval_2016_submit')
    for name in os.listdir('results/test'):
        name, ext = os.path.splitext(name)
        if ext == '.json':
            with open('results/test/{}.json'.format(name), 'r') as in_sr, \
                    open('results/test/{}.tsv'.format(name), 'w') as out_sr:
                for line in in_sr:
                    obj = json.loads(line)
                    out_sr.write('{}\t{}'.format(obj['id'], ['negative', 'neutral', 'positive'][obj['label']]) + '\n')

if __name__ == '__main__':
    main()

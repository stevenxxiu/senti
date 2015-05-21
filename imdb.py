#!/usr/bin/env python

import os
import json

data_files = [os.path.join('data/imdb', path) for path in ('train.json', 'dev.json')]


def to_json():
    labels = {'pos': 10, 'neg': 0, 'unsup': None}
    for s, data_file in zip(('train', 'test'), data_files):
        with open(data_file, 'w') as out_sr:
            for label, score in labels.items():
                path = os.path.join('../code/aclImdb', s, label)
                if not os.path.exists(path):
                    continue
                for name in os.listdir(path):
                    with open(os.path.join(path, name), 'r') as in_sr:
                        contents = in_sr.read()
                        contents = contents.replace('<br />', '\n')
                        out_sr.write('{}\n'.format(json.dumps({contents: score})))


def main():
    to_json()


if __name__ == '__main__':
    main()

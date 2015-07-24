#!/usr/bin/env python

import json
import os


def main():
    os.chdir('data/imdb')
    labels = {'pos': 2, 'neg': 0, 'unsup': None}
    for name in ('train', 'test'):
        with open('{}.json'.format(name), 'w') as out_sr:
            i = 0
            for label, score in labels.items():
                path = os.path.join('../../../code/aclImdb', name, label)
                if not os.path.exists(path):
                    continue
                for file_name in os.listdir(path):
                    with open(os.path.join(path, file_name), 'r') as in_sr:
                        contents = in_sr.read()
                        contents = contents.replace('<br />', '\n')
                        out_sr.write(json.dumps({
                            'id': '{}_{}'.format(name, i), 'text': contents, 'label': score
                        }) + '\n')
                        i += 1


if __name__ == '__main__':
    main()

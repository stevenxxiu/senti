#!/usr/bin/env python

import json
import os


def main():
    os.chdir('data/imdb')
    labels = {'pos': 2, 'neg': 0, 'unsup': None}
    for s, data_file in zip(('train', 'test'), ('train.json', 'dev.json')):
        with open(data_file, 'w') as out_sr:
            i = 0
            for label, score in labels.items():
                path = os.path.join('../../../code/aclImdb', s, label)
                if not os.path.exists(path):
                    continue
                for name in os.listdir(path):
                    with open(os.path.join(path, name), 'r') as in_sr:
                        contents = in_sr.read()
                        contents = contents.replace('<br />', '\n')
                        out_sr.write(json.dumps({'doc_id': str(i), 'text': contents, 'label': score}) + '\n')
                        i += 1


if __name__ == '__main__':
    main()

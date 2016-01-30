#!/usr/bin/env python

import itertools
import json
import os

from senti.rand import *


def write_data(in_path, out_path, labels):
    with open(out_path, 'w') as out_sr:
        for label_name, label in labels.items():
            file_names = os.listdir(os.path.join(in_path, label_name))
            file_names.sort(key=lambda file_name_: int(os.path.splitext(file_name_)[0].split('_', 2)[0]))
            for file_name in file_names:
                with open(os.path.join(in_path, label_name, file_name), 'r', encoding='utf-8') as in_sr:
                    contents = in_sr.read()
                    contents = contents.replace('<br />', '\n')
                    id_, rating = os.path.splitext(file_name)[0].split('_', 2)
                    obj = {'id': '{}_{}'.format(label_name, id_), 'text': contents}
                    if int(rating) != 0:
                        obj['rating'] = int(rating)
                    if label is not None:
                        obj['label'] = label
                    out_sr.write(json.dumps(obj) + '\n')


def split_val(in_path, train_path, val_path, val_split):
    with open(in_path, encoding='utf-8') as in_sr:
        lines = in_sr.readlines()
    get_rng().shuffle(lines)
    lines_iter = iter(lines)
    with open(train_path, 'w') as train_sr, open(val_path, 'w') as val_sr:
        val_sr.write(''.join(itertools.islice(lines_iter, 0, round(len(lines) * val_split))))
        train_sr.write(''.join(lines_iter))


def main():
    os.chdir('data/imdb')
    seed_rng(1234)
    write_data('input/train', 'train.json', {'pos': 2, 'neg': 0})
    write_data('input/test', 'test.json', {'pos': 2, 'neg': 0})
    write_data('input/train', 'unsup.json', {'unsup': None})
    split_val('train.json', 'train.json', 'val.json', 0.1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import os
import itertools

from senti.rand import *


def split_val_test(in_path, train_path, val_path, test_path, val_split, test_split):
    with open(in_path, encoding='utf-8') as in_sr:
        lines = in_sr.readlines()
    get_rng().shuffle(lines)
    lines_iter = iter(lines)
    with open(train_path, 'w') as train_sr, open(val_path, 'w') as val_sr, open(test_path, 'w') as test_sr:
        val_sr.write(''.join(itertools.islice(lines_iter, 0, round(len(lines) * val_split))))
        test_sr.write(''.join(itertools.islice(lines_iter, 0, round(len(lines) * test_split))))
        train_sr.write(''.join(lines_iter))


def main():
    os.chdir('data/yelp')
    seed_rng(1234)
    split_val_test('input/yelp_academic_dataset_review.json', 'train.json', 'val.json', 'test.json', 0.1, 0.1)


if __name__ == '__main__':
    main()

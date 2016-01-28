#!/usr/bin/env python

import os
import pandas as pd
from collections import OrderedDict


def main():
    os.chdir('data/twitter')
    names = []
    for file_name in os.listdir('results/test'):
        name, ext = file_name.split('.', 1)
        if ext == 'txt':
            names.append(name)
    objs = []
    for name in names:
        obj = OrderedDict([('name', name)])
        for test in ('val', 'test'):
            with open('results/{}/{}.txt'.format(test, name)) as sr:
                metrics = dict(zip(['accuracy', 'f1'], [float(line.split(':')[-1]) for line in sr.readlines()[-2:]]))
                obj.update([('{}_{}'.format(test, name), value) for name, value in metrics.items()])
        objs.append(obj)
    df = pd.DataFrame(objs, columns=objs[0].keys())
    df.sort_values('test_f1', ascending=False, inplace=True)
    pd.options.display.max_colwidth = 1000
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()

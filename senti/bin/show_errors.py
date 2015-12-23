#!/usr/bin/env python

import json
import os
from collections import OrderedDict


def main():
    # shows errors sorted by worst probability
    errors = []
    os.chdir('data/twitter')
    with open('results/test/cnn_word(embedding=google).json') as res_sr, open('semeval/test.json', 'r') as gold_sr:
        for res_line, gold_line in zip(res_sr, gold_sr):
            res_obj, gold_obj = json.loads(res_line), json.loads(gold_line)
            if res_obj['label'] != gold_obj['label']:
                errors.append(OrderedDict([
                    ('text', gold_obj['text']),
                    ('pred_label', res_obj['label']), ('gold_label', gold_obj['label']),
                    ('probs', res_obj['probs']),
                ]))
    errors.sort(key=lambda obj_: dict(obj_['probs'])[obj_['gold_label']])
    with open('results/test/cnn_word(embedding=google).errors.json', 'w') as sr:
        for obj in errors:
            sr.write(json.dumps(obj) + '\n')

if __name__ == '__main__':
    main()

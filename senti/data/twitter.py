#!/usr/bin/env python

import html
import json
import os
import re

from senti.features import Emoticons, emoticon_re


def unique(seq):
    seen = set()
    seen_add = seen.add
    return list(x for x in seq if not (x in seen or seen_add(x)))


def main():
    os.chdir('data/twitter')
    class_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    for name, path in [
        ('dev', 'input/unitn/dev/gold/twitter-dev-gold-B.tsv'),
        ('train', 'input/unitn/train/cleansed/twitter-train-cleansed-B.txt')
    ]:
        with open(path) as sr, open('{}.json'.format(name), 'w') as out_sr:
            for line in sr:
                doc_id, label, text = re.match(r'\d+\t(\d+)\t(negative|neutral|positive)\t(.+)', line).groups()
                text = text.encode().decode('unicode-escape')
                text = html.unescape(text)
                out_sr.write(json.dumps({'id': doc_id, 'text': text, 'label': class_map[label]}) + '\n')
    with open('input/test/SemEval2015-task10-test-B-input.txt') as in_sr, \
            open('input/test/SemEval2015-task10-test-B-gold.txt') as labels_sr, open('test.json', 'w') as out_sr:
        for line, label_line in zip(in_sr, labels_sr):
            doc_id, text = re.match(r'NA\t(T\d+)\tunknwn\t(.+)', line).groups()
            text = html.unescape(text)
            doc_id_label, label = re.match(r'\d+\t(T\d+)\t(negative|neutral|positive)', label_line).groups()
            assert doc_id == doc_id_label
            out_sr.write(json.dumps({'id': doc_id, 'text': text, 'label': class_map[label]}) + '\n')
    # with open('input/unsup/unsup.txt', encoding='utf-8') as in_sr, open('unsup.txt', 'w', encoding='utf-8') as out_sr:
    #     for line in in_sr:
    #         out_sr.write(html.unescape(line))
    # full tokenizing is slow, so we use a less complicated re
    emoticon_re_ = re.compile(r'(^|\s)' + emoticon_re.pattern.strip()[1:-1] + r'($|\s)', emoticon_re.flags)
    # all emoticons as cache stage, as emoticons extraction can be very slow
    # with open('unsup.txt', encoding='utf-8') as in_sr, open('emote.txt', 'w') as out_sr:
    #     for i, line in enumerate(in_sr):
    #         if emoticon_re_.search(line):
    #             out_sr.write(line)
    retweet_re = re.compile(r'RT\s*"?[@＠][a-zA-Z0-9_]+:?')
    with open('emote.txt', encoding='utf-8') as in_sr, open('distant.json', 'w') as out_sr:
        for i, line in enumerate(in_sr):
            line = line[:-1]
            if retweet_re.search(line):
                continue
            counts = [0, 0, 0]
            for match in emoticon_re_.finditer(line):
                counts[Emoticons.assess_match(match)] += 1
            label = None
            if counts[0] > 0 and counts[1] == 0 and counts[2] == 0:
                label = 0
            elif counts[0] == 0 and counts[1] == 0 and counts[2] > 0:
                label = 2
            if label is not None:
                out_sr.write(json.dumps({
                    'id': 'unsup_{}'.format(i), 'text': emoticon_re_.sub('', line), 'label': label
                }) + '\n')

if __name__ == '__main__':
    main()

#!/usr/bin/env python

import html
import json
import os
import re
from contextlib import ExitStack, closing
from multiprocessing import Pool

from senti.features import Emoticons
from senti.rand import get_rng


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class UnsupData:
    # full tokenizing is slow, so we use a less complicated re
    emoticon_re = re.compile(r'''
        (?:
            [<>]?
            [:;=8]                     # eyes
            [\-o\*\']?                 # optional nose
            (?P<mouth>[\)\]\(\[dDpP/:\}\{@\|\\]) # mouth
            (\s|$)
        |
            (^|\s)
            (?P<rmouth>[\)\]\(\[dDpP/:\}\{@\|\\]) # mouth
            [\-o\*\']?                 # optional nose
            [:;=8]                     # eyes
            [<>]?
        )
    ''', re.VERBOSE)

    @staticmethod
    def unescape_unsup():
        with open('input/unsup/unsup.txt', encoding='utf-8') as in_sr, \
                open('unsup/all.txt', 'w', encoding='utf-8') as out_sr:
            for line in in_sr:
                out_sr.write(html.unescape(html.unescape(line)))

    @staticmethod
    def _emote_filter(line):
        return bool(UnsupData.emoticon_re.search(line)), line

    @classmethod
    def write_all_emote(cls):
        # all emoticons as cache stage, as emoticons extraction can be very slow
        with open('unsup/all.txt', encoding='utf-8') as in_sr, open('emote/all.txt', 'w') as out_sr:
            with closing(Pool(64)) as pool:
                for res, line in pool.imap(cls._emote_filter, in_sr, 100000):
                    if res:
                        out_sr.write(line)

    @classmethod
    def write_split_emote(cls):
        retweet_re = re.compile(r'RT\s*"?[@＠][a-zA-Z0-9_]+:?')
        with open('emote/all.txt', encoding='utf-8') as in_sr, ExitStack() as stack:
            out_srs = {
                name: stack.enter_context(open('emote/class_{}.txt'.format(name), 'w', encoding='utf-8'))
                for name in ['pos', 'neg']
            }
            for i, line in enumerate(in_sr):
                if retweet_re.search(line):
                    continue
                counts = [0, 0, 0]
                for match in cls.emoticon_re.finditer(line):
                    counts[Emoticons.assess_match(match)] += 1
                label = None
                if counts[0] > 0 and counts[1] == 0 and counts[2] == 0:
                    label = 0
                elif counts[0] == 0 and counts[1] == 0 and counts[2] > 0:
                    label = 2
                if label is not None:
                    out_srs[label].write(cls.emoticon_re.sub(' ', line).strip() + '\n')


class SemEvalData:
    class_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    @classmethod
    def write_unitn(cls, out_path, unitn_path, download_path, is_train):
        with open(unitn_path) as unitn_sr, open(download_path) as download_sr, open(out_path, 'w') as out_sr:
            for unitn_line, download_line in zip(unitn_sr, download_sr):
                doc_id_unitn, label_unitn, text_unitn = \
                    re.match(r'\d+\t(\d+)\t(negative|neutral|positive)\t(.+)', unitn_line).groups()
                doc_id_download, label_download, text_download = \
                    re.match(r'\d+\t(\d+)\t(negative|neutral|positive)\t('r'.+)', download_line).groups()
                text_unitn = text_unitn.encode().decode('unicode-escape')
                text_unitn = text_unitn.replace(r'’', '\'')
                if is_train:
                    text_unitn = html.unescape(text_unitn)
                    text_unitn = text_unitn.replace('""', '"')
                text_download = html.unescape(html.unescape(text_download))
                assert doc_id_unitn == doc_id_download
                assert label_unitn == label_download
                text = text_unitn
                if text_download != 'Not Available':
                    # some differences are impossible to reconcile, some unitn data have the wrong order
                    # if re.sub(r'\s+', ' ', text_unitn) != re.sub(r'\s+', ' ', text_download):
                    #     logging.error(out_path)
                    #     logging.error(text_unitn)
                    #     logging.error(text_download)
                    # assert re.sub(r'\s+', ' ', text_unitn) == re.sub(r'\s+', ' ', text_download)
                    text = text_download
                out_sr.write(json.dumps({'id': doc_id_unitn, 'text': text, 'label': cls.class_map[label_unitn]}) + '\n')

    @classmethod
    def write_test(cls, out_path, download_path, test_path):
        with open(download_path) as in_sr, open(test_path) as labels_sr, open(out_path, 'w') as out_sr:
            for line, label_line in zip(in_sr, labels_sr):
                doc_id, text = re.match(r'NA\t(T\d+)\tunknwn\t(.+)', line).groups()
                text = html.unescape(html.unescape(text))
                doc_id_label, label = re.match(r'\d+\t(T\d+)\t(negative|neutral|positive)', label_line).groups()
                assert doc_id == doc_id_label
                out_sr.write(json.dumps({'id': doc_id, 'text': text, 'label': cls.class_map[label]}) + '\n')


def shuffle_lines(names, in_dir, out_dir):
    lines = []
    splits = []
    for name in names:
        with open(os.path.join(in_dir, name)) as sr:
            cur_lines = list(sr)
            lines.extend(cur_lines)
            splits.append(len(cur_lines))
    get_rng().shuffle(lines)
    offset = 0
    os.makedirs(out_dir)
    for name, split in zip(names, splits):
        with open(os.path.join(out_dir, name), 'w') as sr:
            sr.writelines(lines[offset:offset + split])
            offset += split


def main():
    os.chdir('data/twitter')
    # UnsupData.unescape_unsup()
    # UnsupData.write_all_emote()
    # UnsupData.write_split_emote()
    # for unitn_entry in [(
    #     'semeval/val.json', 'input/unitn/dev/gold/twitter-dev-gold-B.tsv',
    #     'input/dev/gold/twitter-dev-gold-B-downloaded.tsv', False
    # ), (
    #     'semeval/train.json', 'input/unitn/train/cleansed/twitter-train-cleansed-B.txt',
    #     'input/train/cleansed/twitter-train-cleansed-B-downloaded.tsv', True
    # )]:
    #     SemEvalData.write_unitn(*unitn_entry)
    # SemEvalData.write_test(
    #     'semeval/test.json', 'input/test/SemEval2015-task10-test-B-input.txt',
    #     'input/test/SemEval2015-task10-test-B-gold.txt'
    # )
    shuffle_lines(['train.json', 'val.json', 'test.json'], 'semeval', 'semeval_random')


if __name__ == '__main__':
    main()

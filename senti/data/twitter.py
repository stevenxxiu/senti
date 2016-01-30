#!/usr/bin/env python

import html
import json
import os
import re
from contextlib import ExitStack, closing
from multiprocessing import Pool

from senti.features import Emoticons
from senti.rand import *


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
    def write_download(cls, out_path, download_path):
        with open(download_path) as download_sr, open(out_path, 'a+') as out_sr:
            for line in download_sr:
                doc_id, label, text = re.match(r'(?:\d+\t)?(\d+)\t(negative|neutral|positive)\t(.+)', line).groups()
                text = html.unescape(html.unescape(text))
                if text == 'Not Available':
                    continue
                out_sr.write(json.dumps({'id': doc_id, 'text': text, 'label': cls.class_map[label]}) + '\n')

    @classmethod
    def write_unitn(cls, out_path, unitn_path, download_path, is_train):
        with open(unitn_path) as unitn_sr, open(download_path) as download_sr, open(out_path, 'a+') as out_sr:
            for unitn_line, download_line in zip(unitn_sr, download_sr):
                doc_id_unitn, label_unitn, text_unitn = \
                    re.match(r'\d+\t(\d+)\t(negative|neutral|positive)\t(.+)', unitn_line).groups()
                doc_id_download, label_download, text_download = \
                    re.match(r'\d+\t(\d+)\t(negative|neutral|positive)\t(.+)', download_line).groups()
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
    def write_test_2015(cls, out_path, input_path, label_path):
        with open(input_path) as in_sr, open(label_path) as labels_sr, open(out_path, 'a+') as out_sr:
            for line, label_line in zip(in_sr, labels_sr):
                doc_id, text = re.match(r'NA\t(T\d+)\tunknwn\t(.+)', line).groups()
                text = html.unescape(html.unescape(text))
                doc_id_label, label = re.match(r'\d+\t(T\d+)\t(negative|neutral|positive)', label_line).groups()
                assert doc_id == doc_id_label
                out_sr.write(json.dumps({'id': doc_id, 'text': text, 'label': cls.class_map[label]}) + '\n')

    @classmethod
    def write_test_2016(cls, out_path, input_path):
        with open(input_path) as in_sr, open(out_path, 'a+') as out_sr:
            for line in in_sr:
                doc_id, text = re.match(r'(\d+)\tUNKNOWN\t(.+)', line).groups()
                # text = re.sub(r'\\(?!u)', r'\\\\', text)
                text = re.sub(r'\\$', r'\\\\', text)
                text = text.encode().decode('unicode-escape')
                text = html.unescape(html.unescape(text))
                out_sr.write(json.dumps({'id': doc_id, 'text': text}) + '\n')


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
    for name, split in zip(names, splits):
        with open(os.path.join(out_dir, name), 'w') as sr:
            sr.writelines(lines[offset:offset + split])
            offset += split


def create_data_dir(path):
    os.makedirs(path, exist_ok=True)
    for file_name in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_name)):
            os.remove(os.path.join(path, file_name))


def main():
    os.chdir('data/twitter')
    seed_rng(1234)

    # unsup
    # UnsupData.unescape_unsup()
    # UnsupData.write_all_emote()
    # UnsupData.write_split_emote()

    # semeval 2015
    create_data_dir('semeval_2015')
    SemEvalData.write_unitn(
        'semeval_2015/train.json',
        'input/semeval2015_task10_all/unitn/train/cleansed/twitter-train-cleansed-B.txt',
        'input/semeval2015_task10_all/train/cleansed/twitter-train-cleansed-B-downloaded.tsv', True
    )
    SemEvalData.write_unitn(
        'semeval_2015/val.json',
        'input/semeval2015_task10_all/unitn/dev/gold/twitter-dev-gold-B.tsv',
        'input/semeval2015_task10_all/dev/gold/twitter-dev-gold-B-downloaded.tsv', False
    )
    SemEvalData.write_test_2015(
        'semeval_2015/test.json',
        'input/semeval2015_task10_all/test/SemEval2015-task10-test-B-input.txt',
        'input/semeval2015_task10_all/test/SemEval2015-task10-test-B-gold.txt',
    )

    # semeval 2015 random
    create_data_dir('semeval_2015_random')
    shuffle_lines(['train.json', 'val.json', 'test.json'], 'semeval_2015', 'semeval_2015_random')

    # semeval 2016
    create_data_dir('semeval_2016')
    SemEvalData.write_download(
        'semeval_2016/train.json',
        'input/semeval2016-task4.traindev/train/100_topics_100_tweets.sentence-three-point.subtask-A.train.out.txt',
    )
    SemEvalData.write_unitn(
        'semeval_2016/train.json',
        'input/semeval2015_task10_all/unitn/train/cleansed/twitter-train-cleansed-B.txt',
        'input/semeval2015_task10_all/train/cleansed/twitter-train-cleansed-B-downloaded.tsv', True
    )
    SemEvalData.write_unitn(
        'semeval_2016/train.json',
        'input/semeval2015_task10_all/unitn/dev/gold/twitter-dev-gold-B.tsv',
        'input/semeval2015_task10_all/dev/gold/twitter-dev-gold-B-downloaded.tsv', False
    )
    SemEvalData.write_download(
        'semeval_2016/val.json',
        'input/semeval2016-task4.traindev/dev/100_topics_100_tweets.sentence-three-point.subtask-A.dev.out.txt',
    )
    SemEvalData.write_download(
        'semeval_2016/test.json',
        'input/semeval2016-task4.traindev/test/100_topics_100_tweets.sentence-three-point.subtask-A.test.out.txt',
    )

    # semeval 2016 submit
    create_data_dir('semeval_2016_submit')
    SemEvalData.write_download(
        'semeval_2016_submit/train.json',
        'input/semeval2016-task4.traindev/train/100_topics_100_tweets.sentence-three-point.subtask-A.train.out.txt',
    )
    SemEvalData.write_unitn(
        'semeval_2016_submit/train.json',
        'input/semeval2015_task10_all/unitn/train/cleansed/twitter-train-cleansed-B.txt',
        'input/semeval2015_task10_all/train/cleansed/twitter-train-cleansed-B-downloaded.tsv', True
    )
    SemEvalData.write_unitn(
        'semeval_2016_submit/train.json',
        'input/semeval2015_task10_all/unitn/dev/gold/twitter-dev-gold-B.tsv',
        'input/semeval2015_task10_all/dev/gold/twitter-dev-gold-B-downloaded.tsv', False
    )
    SemEvalData.write_download(
        'semeval_2016_submit/train.json',
        'input/semeval2016-task4.traindev/test/100_topics_100_tweets.sentence-three-point.subtask-A.test.out.txt',
    )
    SemEvalData.write_download(
        'semeval_2016_submit/val.json',
        'input/semeval2016-task4.traindev/dev/100_topics_100_tweets.sentence-three-point.subtask-A.dev.out.txt',
    )
    SemEvalData.write_test_2016(
        'semeval_2016_submit/test.json',
        'input/semeval2016_task4_test_datasets/SemEval2016-task4-test.subtask-A.txt',
    )


if __name__ == '__main__':
    main()

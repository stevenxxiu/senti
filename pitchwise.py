#!/usr/bin/env python

import itertools
import json

import lxml.etree
from nltk.tokenize import sent_tokenize


def parse(path):
    res = []
    with open(path, 'r', encoding='utf-8') as sr:
        root = lxml.etree.fromstringlist(itertools.chain('<root>', sr, '</root>'))
        obj = {}
        for node in root:
            tag = node.tag.lower()
            if tag == 'docno':
                obj['docid'] = int(node.text.strip())
            elif tag == 'body':
                obj['text'] = node.text.strip()
                res.append(obj)
                obj = {}
    return res


def to_json(objs, path):
    with open(path, 'w') as sr:
        for obj in objs:
            for i, sent in enumerate(sent_tokenize(obj['text'])):
                json.dump({'docid': obj['docid'], 'sentid': i, 'text': sent}, sr)
                sr.write('\n')


def main():
    to_json(parse('data/pitchwise/test.xml'), 'data/pitchwise/test.json')

if __name__ == '__main__':
    main()

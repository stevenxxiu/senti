
import re

__all__ = ['unescape', 'normalize_special', 'normalize_elongations']


def unescape(text):
    text = text.replace('’', '\'')
    text = text.replace('“', '"')
    text = text.replace('…', '...')
    return text


def normalize_special(text):
    text = re.sub(r'\S{,4}://\S+', ' http://someurl ', text)
    text = re.sub(r'[@＠][a-zA-Z0-9_]+:?', ' @someuser ', text)
    return text


def normalize_elongations(text):
    text = re.sub(r'([a-zA-Z])\{2,}', r'\1\1', text)
    text = re.sub(r'\.{4,}', r'...', text)
    return text

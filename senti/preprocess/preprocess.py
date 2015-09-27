
import re

__all__ = ['unescape', 'normalize']


def unescape(text):
    text = text.replace('’', '\'')
    text = text.replace('“', '"')
    text = text.replace('…', '...')
    return text


def normalize(text):
    text = re.sub(r'\S{,4}://\S+', ' _url ', text)
    text = re.sub(r'[@＠][a-zA-Z0-9_]+:?', ' _user ', text)
    text = re.sub(r'([a-zA-Z])\{2,}', r'\1\1', text)
    text = re.sub(r'\.{4,}', r'...', text)
    return text.strip()

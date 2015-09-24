
import re

__all__ = ['unescape', 'normalize']


def unescape(text):
    text = text.replace('&amp;', '&')
    text = text.replace('\u2019', '\'')
    return text


def normalize(text):
    text = re.sub(r'\S{,4}://\S+', ' _url ', text)
    text = re.sub(r'(RT\s*)?@\w+:?', ' _user ', text)
    text = re.sub(r'([a-z])\1+', r'\1\1', text)
    return text.strip()

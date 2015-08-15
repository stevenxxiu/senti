
import re

__all__ = ['normalize_urls', 'tokenize']


def normalize_urls(text):
    return re.sub(r'\S{,4}://\S+', '_URL', text)


def tokenize(text):
    return re.findall(r'\w+|\$[\d\.]+|\S+', text)

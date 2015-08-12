
import re

__all__ = ['normalize_urls', 'tokenize']


def normalize_urls(text):
    return re.sub(r'\S{,4}://\S+', '_URL', text)


def tokenize(text):
    text = re.sub(r'([\.\",()!?;:])', r' \1 ', text)
    return text.strip().split()

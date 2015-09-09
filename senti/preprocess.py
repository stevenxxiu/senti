
import re

__all__ = ['normalize_urls', 'tokenize', 'tokenize_yookim', 'tokenize_yookim_sst']


def normalize_urls(text):
    return re.sub(r'\S{,4}://\S+', '_URL', text)


def tokenize(text):
    return re.findall(r'\w+|\$[\d\.]+|\S+', text)


def tokenize_yookim(text):
    text = re.sub(r'[^A-Za-z0-9(),!?\'`]', ' ', text)
    text = re.sub(r'\'s', ' \'s', text)
    text = re.sub(r'\'ve', ' \'ve', text)
    text = re.sub(r'n\'t', ' n\'t', text)
    text = re.sub(r'\'re', ' \'re', text)
    text = re.sub(r'\'d', ' \'d', text)
    text = re.sub(r'\'ll', ' \'ll', text)
    text = re.sub(r',', ' , ', text)
    text = re.sub(r'!', ' ! ', text)
    text = re.sub(r'\(', ' \( ', text)
    text = re.sub(r'\)', ' \) ', text)
    text = re.sub(r'\?', ' \? ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().split()


def tokenize_yookim_sst(text):
    text = re.sub(r'[^A-Za-z0-9(),!?\'`]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().split()


import re

__all__ = ['normalize', 'tokenize']


def normalize(text):
    text = re.sub(r'\S{,4}://\S+', '_URL', text)
    text = re.sub(r'\$[\d\.]+', '_MONEY', text)
    text = re.sub(r'(RT\s*)?@\w+:?', '_USER', text)
    return text.strip()


def tokenize(text):
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
    return text.split()

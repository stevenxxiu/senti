
import re

__all__ = ['normalize']


def normalize(text):
    text = text.replace('\u2019', '\'')
    text = re.sub(r'\S{,4}://\S+', '_url', text)
    text = re.sub(r'\$[\d\.]+', '_money', text)
    text = re.sub(r'(RT\s*)?@\w+:?', '_user', text)
    return text.strip()

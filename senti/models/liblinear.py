
import json
import os
import re

from senti.utils import third_dir


class LibLinear:
    def __init__(self):
        self.classes_ = None

    @staticmethod
    def fit(vecs, labels):
        with open('liblinear_train.txt', 'w') as sr:
            for label, vec in zip(labels, vecs):
                line = ' '.join('{}:{}'.format(i + 1, v) for i, v in enumerate(vec))
                sr.write('{} {}\n'.format(json.dumps(label, separators=(',', ':')).replace(' ', '\\u0020'), line))
        os.system('{}/liblinear/train -s 0 liblinear_train.txt liblinear_model.txt'.format(third_dir))

    def predict_proba(self, vecs):
        with open('liblinear_test.txt', 'w') as sr:
            for vec in vecs:
                line = ' '.join('{}:{}'.format(i + 1, v) for i, v in enumerate(vec))
                sr.write('-1 {}\n'.format(line))
        os.system(
            '{}/liblinear/predict -q -b 1 liblinear_test.txt liblinear_model.txt liblinear_predict.txt'
            .format(third_dir)
        )
        with open('liblinear_predict.txt') as sr:
            lines = iter(sr)
            self.classes_ = list(map(json.loads, re.match(r'labels (.+)', next(lines)).group(1).split()))
            for line in lines:
                label, probs = re.match(r'(\S+) (.+)', line).groups()
                yield list(map(float, probs.split()))

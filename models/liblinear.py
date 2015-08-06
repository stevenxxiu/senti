
import itertools
import os
import re

from senti.utils import third_dir


class LibLinear:
    @staticmethod
    def train(labels, vecs):
        with open('liblinear_train.txt', 'w') as sr:
            for label, vec in zip(labels, vecs):
                line = ' '.join('{}:{}'.format(i + 1, v) for i, v in enumerate(vec))
                sr.write('{} {}\n'.format(label, line))
        os.system('{}/liblinear/train -s 0 liblinear_train.txt liblinear_model.txt'.format(third_dir))

    @staticmethod
    def predict(vecs):
        with open('liblinear_test.txt', 'w') as sr:
            for vec in vecs:
                line = ' '.join('{}:{}'.format(i + 1, v) for i, v in enumerate(vec))
                sr.write('-1 {}\n'.format(line))
        os.system(
            '{}/liblinear/predict -b 1 liblinear_test.txt liblinear_model.txt liblinear_predict.txt'
            .format(third_dir)
        )
        with open('liblinear_predict.txt') as sr:
            for line in itertools.islice(sr, 1, None):
                label, probs = re.match(r'(\S+) (.+)', line).groups()
                probs = list(map(float, probs))
                yield label, probs

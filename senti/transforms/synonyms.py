
from nltk.corpus import wordnet as wn
from sklearn.base import BaseEstimator

from senti.rand import get_rng
from senti.utils import reiterable
from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['ReplaceSynonyms']


class ReplaceSynonyms(BaseEstimator, EmptyFitMixin):
    @reiterable
    def transform(self, docs):
        for doc in docs:
            res = [word for word, tag, confidence in doc]
            is_ = get_rng().choice(len(doc), min(len(doc), get_rng().geometric(0.5)), replace=False)
            for i in is_:
                word, tag, confidence = doc[i]
                words = []
                if tag not in 'nvar':
                    continue
                for synset in wn.synsets(word, pos=tag):
                    for lemma in synset.lemma_names():
                        replace_word = lemma.replace('_', ' ')
                        if replace_word.lower() != word.lower():
                            words.append(replace_word)
                word_i = get_rng().geometric(0.5)
                if word_i < len(words):
                    res[i] = words[word_i]
            yield res

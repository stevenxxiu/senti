
from gensim.models import Doc2Vec as Doc2Vec_
from gensim.models import Word2Vec as Word2Vec_
from gensim.utils import SaveLoad as SaveLoad_
from sklearn.base import BaseEstimator

__all__ = ['Word2Vec', 'Doc2Vec']


class SaveLoadPickle:
    @staticmethod
    def save(*args, **kwargs):
        raise ValueError(args, kwargs)

    @classmethod
    def load(cls, model):
        return model

    def __getstate__(self):
        res = self.__dict__.copy()
        try:
            self.save()
        except ValueError as val:
            args, kwargs = val.args
            for key in kwargs['ignore']:
                res.pop(key, None)
        return res

    def __setstate__(self, d):
        self.__dict__ = d
        self.load(self)


class SaveLoadMeta(type):
    '''
    Allows saving & loading via pickle, while reusing the save & load methods.
    '''

    def mro(cls):
        res = super().mro()
        res[res.index(SaveLoad_)] = SaveLoadPickle
        return res


class Word2Vec(Word2Vec_, BaseEstimator, metaclass=SaveLoadMeta):
    fit = Word2Vec_.train


class Doc2Vec(Doc2Vec_, BaseEstimator, metaclass=SaveLoadMeta):
    fit = Doc2Vec_.train

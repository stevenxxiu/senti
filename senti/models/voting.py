
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder

__all__ = ['VotingClassifier']


class VotingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output classification is not supported.')
        if self.voting not in ('soft', 'hard'):
            raise ValueError('Voting must be \'soft\' or \'hard\'; got (voting=%r)' % self.voting)
        if self.weights and len(self.weights) != len(self.estimators):
            raise ValueError(
                'Number of classifiers and weights must be equal; got %d weights, %d estimators'
                % (len(self.weights), len(self.estimators))
            )
        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.estimators_ = []

        for name, clf in self.estimators:
            fitted_clf = clf.fit(X, self.le_.transform(y))
            self.estimators_.append(fitted_clf)

        return self

    def predict(self, X):
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)
        else:
            # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions
            )
        maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X):
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        if self.voting == 'hard':
            raise AttributeError('predict_proba is not available when voting=%r' % self.voting)
        return self._predict_proba

    def transform(self, X):
        if self.voting == 'soft':
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        if not deep:
            return super(VotingClassifier, self).get_params(deep=False)
        else:
            out = super(VotingClassifier, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            for name, step in self.named_estimators.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T

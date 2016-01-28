
import logging
import re
from types import SimpleNamespace

import joblib
import numpy as np
from joblib import Memory
from scipy.optimize import basinhopping
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, LabelEncoder
from sklearn.svm import LinearSVC

from senti.features import *
from senti.models import *
from senti.preprocess import *
from senti.rand import get_rng
from senti.transforms import *
from senti.utils import *
from senti.utils.gensim_ import *
from senti.utils.lasagne_ import *

__all__ = ['SentiModels']


class LazyLabels:
    def __init__(self, labels):
        self.labels = labels
        self.value = None

    def __getitem__(self, item):
        return type(self)(self.labels[item])

    def __call__(self):
        if self.value is None:
            self.value = np.fromiter(self.labels, 'int32')
        return self.value


class SentiModels:
    def __init__(
        self, unsup_docs, distant_docs, distant_labels, train_docs, train_labels, val_docs, val_labels, test_docs
    ):
        self.unsup_docs = unsup_docs
        self.distant_docs = distant_docs
        self.distant_labels = LazyLabels(distant_labels)
        self.train_docs = train_docs
        self.train_labels = LazyLabels(train_labels)
        self.val_docs = val_docs
        self.val_labels = LazyLabels(val_labels)
        self.test_docs = test_docs
        self.memory = Memory(cachedir='cache', verbose=0)

    def fit_voting(self):
        names = [
            # 'svm(word_n_grams,char_n_grams,all_caps,hashtags,punctuations,punctuation_last,emoticons,emoticon_last,'
            # 'elongated,negation_count)',
            # 'logreg(w2v_doc)',
            # 'logreg(w2v_word_avg_google)',
            'word2vec_bayes',
            'cnn_word(embedding=google)',
            'rnn_word(embedding=google)',
        ]
        classifiers = [ExternalModel({
            self.val_docs: 'results/val/{}.json'.format(name), self.test_docs: 'results/test/{}.json'.format(name)
        }) for name in names]
        all_probs = np.array([classifier.predict_proba(self.val_docs) for classifier in classifiers])
        all_probs_first, all_probs_rest = all_probs[0], all_probs[1:]
        label_encoder = LabelEncoder()
        val_label_indexes = label_encoder.fit_transform(self.val_labels())
        # assume w_0=1 as w is invariant to scaling
        w = basinhopping(
            lambda w_: -(val_label_indexes == np.argmax((
                all_probs_first + all_probs_rest * w_.reshape((len(w_), 1, 1))
            ).sum(axis=0), axis=1)).sum(), get_rng().uniform(0, 1, len(classifiers) - 1), niter=1000,
            minimizer_kwargs=dict(method='L-BFGS-B', bounds=[(0, None)] * (len(classifiers) - 1))
        ).x
        w = np.hstack([[1], w])
        w /= w.sum()
        logging.info('w: {}'.format(w))
        estimator = VotingClassifier(list(zip(names, classifiers)), voting='soft', weights=w)
        estimator.estimators_ = classifiers
        return 'vote({})'.format(','.join(names)), estimator

    def fit_svm(self):
        tokenize_raw = CachedFitTransform(Map(compose(tokenize, normalize_special, unescape)), self.memory)
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', tokenize_raw), ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        tokenize_insense = CachedFitTransform(Pipeline([
            ('tokenize', tokenize_sense), ('normalize', MapTokens(str.lower)),
        ]), self.memory)
        features = FeatureUnion([
            ('word_n_grams', FeatureUnion([(n, CachedFitTransform(Pipeline([
                ('tokenize', tokenize_insense),
                ('negation_append', NegationAppend()),
                ('ngrams', WordNGrams(n)),
                ('count', Count()),
                ('binarize', Binarizer(copy=False)),
            ]), self.memory)) for n in range(1, 4 + 1)])),
            ('char_n_grams', FeatureUnion([(n, CachedFitTransform(Pipeline([
                ('tokenize', tokenize_insense),
                ('ngrams', CharNGrams(n)),
                ('count', Count()),
                ('binarize', Binarizer(copy=False)),
            ]), self.memory)) for n in range(3, 5 + 1)])),
            ('all_caps', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', AllCaps()),
                ('count', Count()),
            ])),
            # XXX pos
            ('hashtags', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', HashTags()),
                ('count', Count()),
            ])),
            # XXX lexicons
            ('punctuations', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Punctuations()),
                ('count', Count()),
            ])),
            ('punctuation_last', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Punctuations()),
                ('last', Index(-1)),
            ])),
            ('emoticons', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Emoticons()),
                ('count', Count()),
                ('binarize', Binarizer(copy=False)),
            ])),
            ('emoticon_last', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Emoticons()),
                ('last', Index(-1)),
            ])),
            ('elongated', Pipeline([
                ('tokenize', tokenize_raw),
                ('feature', Elongations()),
                ('count', Count()),
            ])),
            ('negation_count', Pipeline([
                ('tokenize', tokenize_insense),
                ('feature', NegationCount()),
            ])),
        ])
        estimator = Pipeline([('features', features), ('classifier', LinearSVC(C=0.005))])
        estimator.fit(self.train_docs, self.train_labels())
        return 'svm({})'.format(','.join(name for name, _ in features.transformer_list)), estimator

    def fit_logreg(self):
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special, unescape))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        features = FeatureUnion([
            # ('w2v_doc', ToCorporas(Pipeline([
            #     ('tokenize', MapCorporas(tokenize_sense)),
            #     ('feature', MergeSliceCorporas(Doc2VecTransform(CachedFitTransform(Doc2Vec(
            #         dm=0, dbow_words=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20,
            #         workers=16
            #     ), self.memory)))),
            # ]).fit([self.train_docs, self.unsup_docs[:10**6], self.val_docs, self.test_docs]))),
            # ('w2v_word_avg', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecAverage(CachedFitTransform(Word2Vec(
            #         sg=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
            #     ), self.memory))),
            # ]).fit(self.unsup_docs[:10**6])),
            # ('w2v_word_avg_google', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecAverage(joblib.load('../google/GoogleNews-vectors-negative300.pickle'))),
            # ])),
            # ('w2v_word_norm_avg', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecNormAverage(CachedFitTransform(Word2Vec(
            #         sg=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
            #     ), self.memory))),
            # ]).fit(self.unsup_docs[:10**6])),
            ('w2v_word_norm_avg_google', Pipeline([
                ('tokenize', tokenize_sense),
                ('feature', Word2VecNormAverage(joblib.load('../google/GoogleNews-vectors-negative300.pickle'))),
            ])),
            # ('w2v_word_max', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecMax(CachedFitTransform(Word2Vec(
            #         sg=1, size=100, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
            #     ), self.memory))),
            # ]).fit(self.unsup_docs[:10**6])),
            # ('w2v_word_max_google', Pipeline([
            #     ('tokenize', tokenize_sense),
            #     ('feature', Word2VecMax(joblib.load('../google/GoogleNews-vectors-negative300.pickle'))),
            # ])),
            # ('w2v_word_inv', ToCorporas(Pipeline([
            #     ('tokenize', MapCorporas(tokenize_sense)),
            #     ('feature', MergeSliceCorporas(Word2VecInverse(CachedFitTransform(Word2Vec(
            #         sg=1, size=100, window=10, hs=0, negative=5, sample=0, min_count=1, iter=20, workers=16
            #     ), self.memory)))),
            # ]).fit([self.train_docs, self.unsup_docs[:10**5], self.val_docs, self.test_docs]))),
        ])
        classifier = LogisticRegression()
        with temp_log_level({'gensim.models.word2vec': logging.INFO}):
            classifier.fit(features.transform(self.train_docs), self.train_labels())
        estimator = Pipeline([('features', features), ('classifier', classifier)])
        return 'logreg({})'.format(','.join(name for name, _ in features.transformer_list)), estimator

    def fit_word2vec_bayes(self):
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special, unescape))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        estimator = Pipeline([
            ('tokenize', tokenize_sense),
            ('classifier', Word2VecBayes(Word2Vec(
                sg=1, size=100, window=10, hs=1, sample=0, min_count=5, workers=16
            ))),
        ])
        with temp_log_level({'senti.models.word2vec_bayes': logging.INFO, 'gensim.models.word2vec': logging.ERROR}):
            estimator.fit(self.train_docs, self.train_labels())
        return 'word2vec_bayes', estimator

    def _fit_embedding_word(self, embedding_type, construct_docs, tokenize_, d=None):
        if embedding_type == 'google':
            embeddings_ = joblib.load('../google/GoogleNews-vectors-negative300.pickle')
            embeddings_ = SimpleNamespace(X=embeddings_.syn0, vocab={w: v.index for w, v in embeddings_.vocab.items()})
        elif embedding_type == 'twitter':
            estimator = Pipeline([
                ('tokenize', MapCorporas(tokenize_)),
                ('word2vec', MergeSliceCorporas(CachedFitTransform(Word2Vec(
                    sg=1, size=d, window=10, hs=0, negative=5, sample=1e-3, min_count=1, iter=20, workers=16
                ), self.memory))),
            ]).fit([self.train_docs, self.unsup_docs[:10**6], self.val_docs, self.test_docs])
            embeddings_ = estimator.named_steps['word2vec'].estimator
            embeddings_ = SimpleNamespace(X=embeddings_.syn0, vocab={w: v.index for w, v in embeddings_.vocab.items()})
        else:
            embeddings_ = SimpleNamespace(X=np.empty((0, d)), vocab={})
        estimator = Pipeline([
            ('tokenize', MapCorporas(tokenize_)),
            # 0.25 is chosen so the unknown vectors have approximately the same variance as google pre-trained ones
            ('embeddings', MapCorporas(Embeddings(
                embeddings_, rand=lambda shape: get_rng().uniform(-0.25, 0.25, shape).astype('float32'),
                include_zero=True
            ))),
        ])
        estimator.fit(construct_docs)
        return estimator.named_steps['embeddings'].estimator

    @staticmethod
    def _fit_embedding_char(embedding_type, alphabet, d=None):
        if embedding_type == 'onehot':
            X = np.identity(len(alphabet), dtype='float32')
        else:
            X = get_rng().uniform(-0.25, 0.25, (len(alphabet), d)).astype('float32')
        return Embeddings(SimpleNamespace(vocab=dict(zip(alphabet, range(len(alphabet)))), X=X), include_zero=True)

    def fit_nn_word(self):
        distant_docs, distant_labels = self.distant_docs[:2 * 10**5], self.distant_labels[:2 * 10**5]
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        emb_type = 'google'
        emb = self._fit_embedding_word(emb_type, [self.val_docs, self.train_docs, distant_docs], tokenize_sense, d=100)
        ft = Pipeline([
            ('tokenize', tokenize_sense),
            ('embeddings', emb),
        ])
        # cf = CNNWord(
        #     batch_size=64, emb_X=emb.X, input_size=56, conv_param=(100, [3, 4, 5]), dense_params=[],
        #     output_size=3, static_mode=1, max_norm=3, f1_classes=[0, 2]
        # )
        # cf = CNNWordPredInteraction(
        #     batch_size=64, emb_X=emb.X, input_size=56, conv_param=(100, [3, 4, 5]), dense_params=[],
        #     output_size=3, max_norm=3, f1_classes=[0, 2]
        # )
        # cf = RNNWord(batch_size=64, emb_X=emb.X, lstm_param=300, output_size=3, f1_classes=[0, 2])
        cf = RNNMultiWord(
            batch_size=64, input_size=56, emb_X=emb.X, conv_param=3, lstm_param=300, output_size=3, f1_classes=[0, 2]
        )
        kw = dict(val_docs=ft.transform(self.val_docs), val_y=self.val_labels())
        cf.fit(ft.transform(distant_docs), distant_labels(), epoch_size=10**4, max_epochs=20, **kw)
        cf.fit(ft.transform(self.train_docs), self.train_labels(), epoch_size=1000, max_epochs=100, **kw)
        estimator = Pipeline([('features', ft), ('classifier', cf)])
        return '{}(embedding={})'.format(snake_case(type(cf).__name__), emb_type), estimator

    def fit_cnn_char(self):
        distant_docs, distant_labels = self.distant_docs[:10**6], self.distant_labels[:10**6]
        normalize = Map(compose(str.lower, str.strip, lambda s: re.sub(r'\s+', ' ', s), normalize_special))
        alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
        emb = self._fit_embedding_char('onehot', alphabet)
        ft = Pipeline([
            ('normalize', normalize),
            ('embeddings', emb),
        ])
        ft_syn = Pipeline([
            ('pos', CachedFitTransform(ArkTweetPosTagger(), self.memory)),
            ('pos_map', MapTokens(lambda entry: (entry[0], {
                'N': 'n', 'V': 'v', 'A': 'a', 'R': 'r',
            }.get(entry[1], 'o'), entry[2]))),
            ('syn', ReplaceSynonyms()),
            ('normalize', MapTokens(normalize_special)),
            ('embeddings', emb),
        ])
        ft_typo = Pipeline([
            ('normalize', normalize),
            ('typos', IntroduceTypos(alphabet)),
            ('embeddings', emb),
        ])
        cf = CNNChar(batch_size=128, emb_X=emb.X, input_size=140, output_size=3, static_mode=0, f1_classes=[0, 2])
        # cf = CachedFitTransform(cf, self.memory)
        kw = dict(val_docs=ft.transform(self.val_docs), val_y=self.val_labels())
        cf.fit(ft.transform(distant_docs), distant_labels(), epoch_size=10**4, max_epochs=100, **kw)
        # cf = NNShallow(batch_size=128, model=classifier, num_train=5)
        cf.fit(ft_typo.transform(self.train_docs), self.train_labels(), max_epochs=15, **kw)
        estimator = Pipeline([('features', ft), ('classifier', cf)])
        return 'cnn_char', estimator

    def fit_multiview_cnn_word_cnn_char(self):
        distant_docs, distant_labels = self.distant_docs[:10**6], self.distant_labels[:10**6]
        # word
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        emb_type = 'google'
        emb_word = self._fit_embedding_word(
            emb_type, [self.val_docs, self.train_docs, distant_docs], tokenize_sense, d=100
        )
        # char
        normalize = Map(compose(str.lower, str.strip, lambda s: re.sub(r'\s+', ' ', s), normalize_special))
        alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
        emb_char = self._fit_embedding_char('onehot', alphabet)
        ft_word = Pipeline([
            ('tokenize', tokenize_sense),
            ('embeddings', emb_word),
        ])
        ft_char = Pipeline([
            ('normalize', normalize),
            ('embeddings', emb_char),
        ])
        ft_char_typo = Pipeline([
            ('normalize', normalize),
            ('typos', IntroduceTypos(alphabet)),
            ('embeddings', emb_char),
        ])
        ft = Zip(ft_word, ft_char)
        ft_typo = Zip(ft_word, ft_char_typo)
        # model
        cf = NNMultiView(models_=[CNNWord(
            batch_size=128, emb_X=emb_word.X, input_size=56, conv_param=(100, [3, 4, 5]), dense_params=[],
            output_size=3, static_mode=1, max_norm=3, f1_classes=[0, 2]
        ), CNNChar(
            batch_size=128, emb_X=emb_char.X, input_size=140, output_size=3, static_mode=1, f1_classes=[0, 2]
        )], output_size=3)
        kw = dict(val_docs=ft.transform(self.val_docs), val_y=self.val_labels(), average_classes=[0, 2])
        cf.fit(ft.transform(distant_docs), distant_labels(), epoch_size=10**4, max_epochs=100, **kw)
        cf.fit(ft_typo.transform(self.train_docs), self.train_labels(), max_epochs=10, **kw)
        estimator = Pipeline([('features', ft), ('classifier', cf)])
        return 'multiview_cnn_word_cnn_char(embedding={})'.format(emb_type), estimator

    def _fit_rnn_embedding(self):
        emb_word = joblib.load('../google/GoogleNews-vectors-negative300.pickle')
        alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
        emb_char = self._fit_embedding_char('none', alphabet, 300)
        ft_char = Pipeline([
            ('normalize', Map(str.lower)),
            ('embeddings', emb_char),
        ])
        # sd = np.mean((emb_word.syn0 - np.mean(emb_word.syn0))**2)**0.5
        cf_char = RNNCharToWordEmbedding(
            batch_size=128, emb_X=emb_char.X, lstm_params=(300, 300), output_size=emb_char.X.shape[1]
        )
        cf_char.fit(
            ft_char.transform(emb_word.vocab), emb_word.syn0, epoch_size=10**3, max_epochs=2 * 300,
            update_params_iter=geometric_learning_rates(0.01, 0.5, 10)
        )
        return cf_char

    def fit_rnn_char_cnn_word(self):
        distant_docs, distant_labels = self.distant_docs[:10**6], self.distant_labels[:10**6]
        alphabet = ' abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
        emb_char = self._fit_embedding_char('none', alphabet, 300)
        ft_char = Pipeline([
            ('normalize', Map(str.lower)),
            ('add_space', Map(lambda s: s + ' ')),
            ('embeddings', emb_char),
        ])
        ft_char_typo = Pipeline([
            ('normalize', Map(str.lower)),
            ('typos', IntroduceTypos(alphabet, 0.9)),
            ('add_space', Map(lambda s: s + ' ')),
            ('embeddings', emb_char),
        ])
        tokenize_sense = CachedFitTransform(Pipeline([
            ('tokenize', Map(compose(tokenize, normalize_special))),
            ('normalize', MapTokens(normalize_elongations)),
        ]), self.memory)
        ft_word = Pipeline([
            ('tokenize', tokenize_sense),
            ('char', MapCorporas(ft_char)),
        ])
        ft_word_typo = Pipeline([
            ('tokenize', tokenize_sense),
            ('char', MapCorporas(ft_char_typo)),
        ])
        cf = RNNCharCNNWord(
            batch_size=64, emb_X=emb_char.X, num_words=56, lstm_params=[300], conv_param=(100, [3, 4, 5]),
            output_size=3, f1_classes=[0, 2]
        )
        kw = dict(val_docs=ft_word.transform(self.val_docs), val_y=self.val_labels())
        cf.fit(
            ft_word.transform(distant_docs), distant_labels(), epoch_size=10**4, max_epochs=100,
            save_best=False, **kw
        )
        cf.fit(ft_word_typo.transform(self.train_docs), self.train_labels(), max_epochs=15, **kw)
        estimator = Pipeline([('features', ft_word), ('classifier', cf)])
        return 'rnn_char_cnn_word', estimator

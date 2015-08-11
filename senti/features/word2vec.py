
from sklearn.base import BaseEstimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

__all__ = ['Word2VecDocs', 'Word2VecWordAverage', 'Word2VecWordMax', 'Word2VecInverse']


class Word2VecDocs(BaseEstimator):
    def __init__(self, preprocessor, tokenizer, dev_docs, unsup_docs=(), *args, **kwargs):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.train_docs = ()
        self.unsup_docs = unsup_docs
        self.dev_docs = dev_docs
        self.doc2vec = Doc2Vec(*args, **kwargs)
        self._train_docs_end = 0
        self._unsup_docs_end = 0
        self._dev_docs_end = 0

    def _all_docs(self):
        i = -1
        for i, doc in enumerate(self.train_docs):
            yield TaggedDocument(self.tokenizer(self.preprocessor(doc)), (i,))
        self._train_docs_end = i + 1
        i = -1
        for i, doc in enumerate(self.unsup_docs):
            yield TaggedDocument(self.tokenizer(self.preprocessor(doc)), (self._train_docs_end + i,))
        self._unsup_docs_end = self._train_docs_end + i + 1
        i = -1
        for i, doc in enumerate(self.dev_docs):
            yield TaggedDocument(self.tokenizer(self.preprocessor(doc)), (self._unsup_docs_end + i,))
        self._dev_docs_end = self._unsup_docs_end + i + 1

    def fit(self, docs, y):
        self.train_docs = docs
        all_docs = list(self._all_docs())
        self.doc2vec.build_vocab(all_docs)
        self.doc2vec.train(all_docs)
        return self

    def transform(self, docs):
        if docs == self.train_docs:
            return self.doc2vec.docvecs[range(self._train_docs_end)]
        elif docs == self.dev_docs:
            return self.doc2vec.docvecs[range(self._unsup_docs_end, self._dev_docs_end)]
        else:
            raise ValueError('docs were not fitted')


class Word2VecWords(BaseEstimator):
    pass

#     def __init__(self, name, src_sr, cmd=None, reuse=False):
#         self.w2v_sr = Word2Vec(src_sr, cmd, reuse)
#         super().__init__(name, (src_sr,), reuse, self.w2v_sr.reuse_options)
#
#     def get_words(self):
#         word_to_vecs = {}
#         lines = iter(self.w2v_sr)
#         dims = tuple(map(int, re.match(r'(\d+) (\d+)', next(lines)).groups()))
#         for line in lines:
#             word, vec = re.match(r'(\S+) (.+)', line).groups()
#             word_to_vecs[word] = np.fromiter(vec.strip().split(), float)
#         return dims, word_to_vecs
#

class Word2VecWordAverage(BaseEstimator):
    pass

#     def __init__(self, src_sr, cmd=None, reuse=False):
#         super().__init__('word2vec_avg({})'.format(src_sr.name), src_sr, cmd, reuse)
#
#     def _iter(self):
#         dims, word_to_vecs = self.get_words()
#         for obj in self.src_srs[0]:
#             words = obj.pop('tokens')
#             obj['vec'] = np.vstack(word_to_vecs[word] for word in words if word in word_to_vecs).mean(0)
#             yield obj
#

class Word2VecWordMax(BaseEstimator):
    pass

#     '''
#     Component-wise abs max. Doesn't make much sense because the components aren't importance, but worth a try.
#     '''
#
#     def __init__(self, src_sr, cmd=None, reuse=False):
#         super().__init__('word2vec_max({})'.format(src_sr.name), src_sr, cmd, reuse)
#
#     def _iter(self):
#         dims, word_to_vecs = self.get_words()
#         for obj in self.src_srs[0]:
#             words = obj.pop('tokens')
#             words_matrix = np.vstack(word_to_vecs[word] for word in words if word in word_to_vecs)
#             arg_maxes = abs(words_matrix).argmax(0)
#             obj['vec'] = words_matrix[arg_maxes, np.arange(len(arg_maxes))]
#             yield obj
#

class Word2VecInverse(BaseEstimator):
    pass
#     '''
#     Performs a document embedding.
#     '''
#
#     def __init__(self, src_sr, cmd=None, reuse=False):
#         self.cmd = cmd or '''
#             -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1
#             -sentence-vectors 1
#         '''.replace('\n', '')
#         super().__init__('word2vec_inv({})'.format(src_sr.name), (src_sr,), reuse, self.cmd)
#         self.reuse_name = '{}.txt'.format(self.name)
#
#     def _iter(self):
#         # number documents using integers for easy sorting later
#         word_to_docs = defaultdict(set)
#         for i, obj in enumerate(self.src_srs[0]):
#             words = obj['tokens']
#             for word in words:
#                 word_to_docs[word].add(str(i))
#         with open('word2vec_train.txt', 'w', encoding='utf-8') as out_sr:
#             for word, docs in word_to_docs.items():
#                 # remove common & rare words
#                 if 10 <= len(docs) <= 2000:
#                     out_sr.write('_*{} {}\n'.format(word, ' '.join(docs)))
#         os.system(
#             'time {}/word2vec/word2vec -train word2vec_train.txt -output {} {}'
#             .format(third_dir, shlex.quote(self.reuse_name), self.cmd)
#         )
#         with open(self.reuse_name, encoding='ISO-8859-1') as in_sr, open('{}~'.format(self.reuse_name), 'w') as out_sr:
#             for line in itertools.islice(in_sr, 1, None):
#                 if not line.startswith('_*') and not line.startswith('</s>'):
#                     out_sr.write(line)
#         os.system('sort -n {0}~ > {0}'.format(shlex.quote(self.reuse_name)))
#         os.system('rm {}~'.format(shlex.quote(self.reuse_name)))
#
#     def __iter__(self):
#         if not self.reusable():
#             self._iter()
#             with open(self.options_name, 'w') as sr:
#                 json.dump(self.reuse_options, sr)
#         with open(self.reuse_name) as sr:
#             for obj, line in zip(self.src_srs[0], sr):
#                 vec = re.match(r'(\S+) (.+)', line).group(2)
#                 obj['vec'] = np.fromiter(vec.strip().split(), float)
#                 yield obj

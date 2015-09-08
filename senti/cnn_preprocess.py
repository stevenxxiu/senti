
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

from senti.features.word2vec import Word2Vec


# XXX cnn is special for now


def build_data_cv(file, cv=10):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    classes = defaultdict(list)
    with open(file, "r") as f:
        for line in f:
            obj = json.loads(line)
            text = clean_str(obj['text'].strip())
            words = set(text.split())
            for word in words:
                vocab[word] += 1
            classes[obj['label']].append(text)
    # split into folds
    for label, texts in classes.items():
        for text in texts:
            revs.append({"y": label, "text": text, "num_words": len(text.split()), "split": np.random.randint(0, cv)})
    return revs, vocab


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def main():
    os.chdir('../data/twitter')
    print("loading data...",)
    revs, vocab = build_data_cv('train.json', cv=10)
    max_l = max(rev["num_words"] for rev in revs)
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...",)
    w2v_obj = Word2Vec()
    w2v_obj.load_binary('../google/GoogleNews-vectors-negative300.bin')
    w2v = dict((word, w2v_obj.X[i]) for word, i in w2v_obj.words.items() if word in vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    pickle.dump([revs, W, W2, word_idx_map, vocab], open("cnn.pickle", "wb"))
    print("dataset created!")

if __name__ == '__main__':
    main()

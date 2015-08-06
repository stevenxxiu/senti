#!/usr/bin/env python

from senti.features.word2vec import *
from senti.models.liblinear import LibLinear
from senti.preprocess import NormTextStream
from senti.stream import MergedStream, SourceStream


def main():
    os.chdir('data/twitter')
    data_files = ['train.json', 'dev.json', 'test.json']

    # throw all files into word2vec
    normed_sr = NormTextStream(MergedStream(list(map(SourceStream, data_files))))
    w2v_doc_sr = Word2VecDocs(normed_sr, reuse=True)
    w2v_word_avg_sr = Word2VecWordMax(normed_sr, reuse=True)
    w2v_word_max_sr = Word2VecWordAverage(normed_sr, reuse=True)
    w2v_word_inv_sr = Word2VecInverse(normed_sr, reuse=True)

    # train & write results
    vecs = (obj['vec'] for obj in w2v_word_inv_sr)
    LibLinear.train((obj['label'] for obj in SourceStream('train.json')), vecs)
    with open('w2v_word_inv_sr.results.json', 'w') as sr:
        for src_obj, (label, probs) in zip(SourceStream('train.json'), LibLinear.predict(vecs)):
            sr.write(json.dumps({
                'id': src_obj['id'], 'label': label,
                'prob_neg': probs['0'], 'prob_nt': probs['1'], 'prob_pos': probs['2']
            }) + '\n')


if __name__ == '__main__':
    main()

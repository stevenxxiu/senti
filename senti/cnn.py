
import pickle
import numpy as np
import os

from senti.models.cnn import *


def main():
    # XXX cnn is special for now
    os.chdir('../data/twitter')
    x = pickle.load(open('cnn.pickle', 'rb'))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print('data loaded!')
    word_vectors = 'rand'
    if word_vectors == 'rand':
        print('using: random vectors')
        U = W2
    elif word_vectors == 'word2vec':
        print('using: word2vec vectors')
        U = W
    results = []
    r = range(0,10)
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=56,k=300, filter_h=5)
        perf = train_conv_net(
            datasets, U, lr_decay=0.95, img_w=300, filter_hs=[3,4,5], conv_non_linear='relu', hidden_units=[100,3],
            shuffle_batch=True, n_epochs=25, sqr_norm_lim=9, non_static=True, batch_size=50, dropout_rate=[0.5]
        )
        print('cv: ' + str(i) + 'perf: ' + str(perf))
        results.append(perf)
    print(str(np.mean(results)))

if __name__ == '__main__':
    main()

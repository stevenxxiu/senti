
import numpy as np
from keras.constraints import *
from keras.layers.convolutional import *
from keras.layers.core import *
from keras.layers.embeddings import *

from senti.utils.keras_ import *

__all__ = ['CNNWord']


class CNNWord(Graph):
    def __init__(
        self, batch_size, emb_X, input_size, conv_param, dense_params, output_size, static_mode, norm_lim, f1_classes
    ):
        super().__init__(batch_size)
        self.input_size = input_size
        self.add_input(name='docs', batch_input_shape=(self.batch_size, input_size), dtype='int')
        embeds = []
        if static_mode in (0, 2):
            self.add_node(Embedding(
                *emb_X.shape, input_length=input_size, weights=[emb_X], W_constraint=identity
            ), input='docs')
            embeds.append(self._prev_name)
        if static_mode in (1, 2):
            self.add_node(Embedding(*emb_X.shape, input_length=input_size, weights=[emb_X]), input='docs')
            embeds.append(self._prev_name)
        convs = []
        for filter_length in conv_param[1]:
            curs = []
            for embed in embeds:
                self.add_node(ZeroPadding1D(filter_length - 1), input=embed)
                self.add_node(Convolution1D(conv_param[0], filter_length, activation='relu'))
                curs.append(self._prev_name)
            self.add_node(MaskedLayer(), inputs=curs, merge_mode='sum')
            self.add_node(MaxPooling1D(input_size + filter_length - 1))
            self.add_node(Flatten())
            convs.append(self._prev_name)
        self.add_node(MaskedLayer(), inputs=convs, merge_mode='concat')
        self.add_node(Dropout(0.5))
        for dense_param in dense_params:
            self.add_node(Dense(dense_param, activation='relu', W_constraint=maxnorm(norm_lim)))
            self.add_node(Dropout(0.5))
        self.add_node(Dense(output_size, activation=log_softmax, W_constraint=maxnorm(norm_lim)))
        self.add_node(LambdaTest(lambda x: T.exp(x)))
        self.add_output('output')
        self.compile(
            'adadelta', {'output': categorical_crossentropy_exp},
            output_ndim={'output': 1}, output_dtype={'output': 'int'},
            train_metrics={'output': ['acc']}, test_metrics={'output': ['acc']}
        )

    def _pre_transform(self, data):
        return {
            'docs': np.vstack(np.pad(
                doc[:self.input_size], (0, max(self.input_size - doc.size, 0)), 'constant'
            ) for doc in data['docs']),
            'output': np.array(data['output']),
        }

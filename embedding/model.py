'''
This module contains a simple wrapper around Tensorflow graph.
It is used to embed sequences of texts
'''
import numpy as np
import tensorflow as tf
import pickle

from keras.preprocessing.sequence import pad_sequences

from graph import PREFIX, load


# TODO: put this in configuration file
INPUT_TENSOR_NAME = 'embedding_1_input:0'
OUTPUT_TENSOR_NAME = 'embedding/Relu:0'
LEARNING_PAHSE = 'dropout_1/keras_learning_phase:0'


class Model(object):
    '''
    Wrapper to embed text using trained model and tokenizer
    '''

    def __init__(self, frozen_graph_filename, tokenizer_path, maxlen):
        print('Loading the graph')
        self.graph = load(frozen_graph_filename)
        self.X = self.graph.get_tensor_by_name("%s/%s" % (PREFIX, INPUT_TENSOR_NAME))
        self.Y = self.graph.get_tensor_by_name("%s/%s" % (PREFIX, OUTPUT_TENSOR_NAME))
        self.LF = self.graph.get_tensor_by_name("%s/%s" % (PREFIX, LEARNING_PAHSE))

        self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        self.persistent_sess = tf.Session(graph=self.graph)

        self.maxlen = maxlen

    def embed(self, texts):
        ''' use model to find prediction '''
        # our graph expect tensors of shape '(?, 1)'
        if not isinstance(texts, (list, tuple)):
            texts = [texts]

        x = self.tokenizer.texts_to_sequences(texts)
        x = pad_sequences(x, maxlen=self.maxlen, padding='post', truncating='post')
        x = np.array(x).reshape(-1, self.maxlen)
        h = self.persistent_sess.run(self.Y, feed_dict={
            self.X: x,
            self.LF: False
        })

        return h

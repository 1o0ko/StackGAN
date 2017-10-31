'''
This module contains a simple wrapper around Tensorflow graph. It is used to embed sequences of texts
'''
import os
import sys

import numpy as np
import tensorflow as tf
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from graph import PREFIX, load
#TODO: put this in configuration file 
INPUT_TENSOR_NAME='embedding_1_input:0'
OUTPUT_TENSOR_NAME='embedding/Relu:0'
LEARNING_PAHSE='dropout_1/keras_learning_phase:0'


class Model(object):
    '''
    Wrapper to embed text using trained model and tokenizer
    '''
    def __init__(self, frozen_graph_filename, tokenizer_path):
        print('Loading the graph')
        graph = load(frozen_graph_filename)
        self.X = graph.get_tensor_by_name("%s/%s" % (PREFIX, INPUT_TENSOR_NAME))
        self.Y = graph.get_tensor_by_name("%s/%s" % (PREFIX, OUTPUT_TENSOR_NAME))
        self.LF = graph.get_tensor_by_name("%s/%s" % (PREFIX, LEARNING_PAHSE))

        self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        self.persistent_sess = tf.Session(graph=graph)

        # load the max sentence padding from input tensor shape
        self.maxlen = self.X.shape[1].value

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
            self.LF:  False
        })

        return h

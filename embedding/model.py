'''
This module contains a simple wrapper around Tensorflow graph.
It is used to embed sequences of texts
'''
import numpy as np
import tensorflow as tf
import pickle

from keras.preprocessing.sequence import pad_sequences

from graph import PREFIX, load


INPUT_TENSOR_NAME = 'embedding_1_input:0'
OUTPUT_TENSOR_NAME = 'embedding/Relu:0'
LEARNING_PAHSE = 'dropout_1/keras_learning_phase:0'

# TODO: add pipeline class that is serializable and shared between Model class
#       and `train_embedding_model.script`


class Model(object):
    '''
    Wrapper to embed text using trained model and tokenizer
    '''

    def __init__(self, frozen_graph_filename, tokenizer_path, maxlen,
                 input_tensor_name=INPUT_TENSOR_NAME,
                 output_tensor_name=OUTPUT_TENSOR_NAME,
                 learning_phase=LEARNING_PAHSE):
        print('Loading the graph')
        self.graph = load(frozen_graph_filename)
        self.X = self.graph.get_tensor_by_name("%s/%s" % (PREFIX, input_tensor_name))
        self.Y = self.graph.get_tensor_by_name("%s/%s" % (PREFIX, output_tensor_name))
        self.LF = self.graph.get_tensor_by_name("%s/%s" % (PREFIX, learning_phase))

        self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        self.persistent_sess = tf.Session(graph=self.graph)

        self.maxlen = maxlen

    def embed(self, texts):
        ''' use model to find prediction '''
        # our graph expect tensors of shape '(?, 1)'
        if not isinstance(texts, (list, tuple)):
            texts = [texts]

        sequences = self.tokenizer.texts_to_sequences(texts)

        # Padding data
        data = pad_sequences(
            sequences,
            maxlen=self.maxlen,
            padding='post',
            truncating='post')

        data = np.array(data).reshape(-1, self.maxlen)

        # feed data through graph
        h = self.persistent_sess.run(self.Y, feed_dict={
            self.X: data,
            self.LF: False
        })

        return h

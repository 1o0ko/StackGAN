"""
Usage: example.py DATA_PATH GLOVE OUTPUT_PATH [options]

Arguments:
    DATA_PATH      path with data
    OUTPUT_PATH    path to save the model file
    GLOVE          file where glove vectors are saved

Options:
    -e, --epochs=<int>           Limit on the number of parsed lines
                                 [default: 10]
    -w, --words=<int>            Maximum number of words in dictionary
                                 [default: 7000]
    -s, --sent-length=<int>      Maximum number of words in the sentence
                                 [default:400]
"""
import os
import logging
import string

import keras
import numpy as np
import tensorflow as tf

from collections import Counter, namedtuple

from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.utils.np_utils import to_categorical

from typeopt import Arguments

BLACK_LIST = string.punctuation.replace('%', '').replace('-', '') + '\n'


def normalize(text, black_list=BLACK_LIST, vocab=None,
              lowercase=True, tokenize=False):
    if black_list:
        text = text.translate(None, BLACK_LIST)
    if lowercase:
        text = text.lower()
    if vocab:
        text = ' '.join([word for word in text.split() if word in vocab])
    if tokenize:
        return text.split()
    return text


def load_and_process(data_path, num_words, maxlen):
    with open(data_path, 'rt') as f:
        classes, texts = zip(*[line.split(" ", 1) for line in f.readlines()])

        # class preprocessing
        classes = [cls[9:] for cls in classes]
        class_to_id = {
            key: index for (index, (key, value)) in enumerate(Counter(classes).most_common())
        }
        ids = to_categorical([class_to_id[cls] for cls in classes])

    # Setting up keras tokenzer
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    logger.debug('Found %s unique tokens', len(word_index))

    # Padding data
    data = pad_sequences(
        sequences,
        maxlen=maxlen,
        padding='post',
        truncating='post')

    logger.debug('Shape of data tensor: %s', data.shape)
    logger.debug('Shape of label tensor: %s', ids.shape)

    return data, ids, word_index


def load_glove_embeddings(embedding_path, word_index,
                          max_sequence, trainable=True):
    '''
    Loads Glove word vectors
    Arguments:
        embedding_path  - path to GloVe word embeddings
        word_index      - dictionary mapping words to their rank
    '''
    logger = logging.getLogger(__name__)

    # create dictionary with embeddings
    embeddings_index = {}
    with open(embedding_path) as f:
        for line in f:
            word, coefs = line.split(" ", 1)
            coefs = np.asarray(coefs.split(), dtype='float32')
            embeddings_index[word] = coefs

    logger.debug('Found %s word vectors with shape', len(embeddings_index))

    # for convenience
    nrows, ncols = len(word_index) + 1, coefs.shape[0]
    logger.debug("rows %s, columns %s", nrows, ncols)

    # words not found in embedding index will be all-zeros
    embedding_matrix = np.zeros((nrows, ncols))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(nrows,
                                ncols,
                                weights=[embedding_matrix],
                                input_length=max_sequence,
                                trainable=trainable)
    return embedding_layer


def train_val_split(data, labels, split_ratio, seed=0):
    '''
    Splits data and lables into training and validation set
    '''
    # set seed
    np.random.seed(seed)

    # shuffle indices
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(split_ratio * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]

    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    return x_train, y_train, x_val, y_val


def build_model(word_index, glove_path, max_sent):
    logger = logging.getLogger(__name__)

    logger.debug('Loading glove embeddings')
    embedding_layer = load_glove_embeddings(glove_path, word_index, max_sent)

    logger.debug('Building model')
    model = Sequential()
    model.add(embedding_layer)

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(5))

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())

    # add dropout
    model.add(Dropout(dropout_rate))

    # add l2 regularization
    model.add(
        Dense(
            1024,
            name="embedding",
            activation='relu',
            kernel_regularizer=l2(.01)))
    model.add(Dense(nb_classes, activation='softmax'))

    # Setup optimizer
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # Compile model
    logger.debug('Compiling the model')
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    return model


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # load and process data
    data, labels, word_index = load_and_process(
        args.data_path, args.words, args.sent_length
    )

    # make split
    x_train, y_train, x_val, y_val = train_val_split(data, labels, 0.1)

    # Build and train a model
    model = build_model(word_index, args.glove, args.sent_length)
    model.summary()
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=128,
        verbose=1)

    # evalute model on train data to see how well we're fitting the data
    logger.info("Train data")
    logger.info(model.evaluate(x_train, y_train, batch_size=128))

    # evalute model on validation data
    logger.info("Validation data")
    logger.info(model.evaluate(x_val, y_val, batch_size=128))

    # all new operations will be in test mode from now on (dropout, etc.)
    K.set_learning_phase(0)
    saver = tf.train.Saver()

    with K.get_session() as ses:
        saver.save(sess, os.path.join(args.output_path, 'model'))


if __name__ == '__main__':
    args = Arguments(__doc__, version='0.1')
    main(args)

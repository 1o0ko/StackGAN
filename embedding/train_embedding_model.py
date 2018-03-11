"""
Usage: example.py DATA_PATH GLOVE OUTPUT_PATH [options]

Arguments:
    DATA_PATH      path with data
    OUTPUT_PATH    path to save the model file
    GLOVE          file where glove vectors are saved

Options:
    -w, --words=<int>            Maximum number of words in dictionary
                                 [default: 10000]
    -s, --sent-length=<int>      Maximum number of words in the sentence
                                 [default: 70]

    -t, --test-size=<float>      Percentage size of the test data
                                 [default: 0.1]

    -m, --min-count=<int>        Minimum class count
                                 [default: 2]
    -d, --dropout=<float>        Dropout rate
                                 [default: 0.2]

    -e, --epochs=<int>           Limit on the number of parsed lines
                                 [default: 20]
    -b, --batch-size=<int>       Size of the batch used for training
                                 [default: 128]

    -v, --verbose                Boolean flag setting the amout of logging

    --early-stopping-patience=<int> Wait 'n' epochs before stopping
                                    [default: 4]
"""

import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',)

import string
import pickle

import numpy as np
import tensorflow as tf

from collections import Counter

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import StratifiedShuffleSplit

from docopt import docopt

BLACK_LIST = string.punctuation.replace('%', '').replace('-', '') + '\n'

from preprocessing import normalize


def get_categorical_accuracy_keras(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)))


def load_data(data_path, verbose=False):
    logger = logging.getLogger(__name__)
    with open(data_path, 'rt') as f:
        classes, texts = zip(*[line.split(" ", 1) for line in f.readlines()])

        # class preprocessing
        classes_stats = Counter(classes).most_common()
        class_to_id = {
            key: index for (index, (key, value)) in enumerate(classes_stats)
        }
        labels = to_categorical([class_to_id[cls] for cls in classes])

        if verbose:
            logger.info("Class statistics")
            logger.info("Found %i classes" % len(classes_stats))
            for key, value in classes_stats:
                logger.info("\t %s: %i" % (key, value))

    return texts, labels


def process_data(texts, num_words, maxlen):
    logger = logging.getLogger(__name__)

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

    return data, tokenizer


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


def get_split(data, labels, test_size, min_count=2, seed=0):
    # we want to filter classes less frequent than 2
    class_info = np.argmax(labels, axis=1)
    class_counts = Counter(class_info)
    labels_, data_ = zip(*[
        (cls, txt) for cls, txt in zip(labels, data) if class_counts[np.argmax(cls)] >= min_count]
    )

    # due to high class impalance whe need to stratify the split
    labels_ = np.array(labels_)
    data_ = np.array(data_)
    class_info = np.argmax(labels_, axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = list(sss.split(class_info, class_info))[0]

    return train_idx, test_idx, labels_, data_


def train_val_split(data, labels, test_size, min_count=2, seed=0):
    '''
    Splits data and lables into training and validation set
    '''
    train_idx, test_idx, labels_, data_ = get_split(
        data, labels, test_size, min_count, seed)

    x_train = data_[train_idx]
    y_train = labels_[train_idx]

    x_val = data_[test_idx]
    y_val = labels_[test_idx]

    return x_train, y_train, x_val, y_val


def build_model(word_index, glove_path, max_sent, dropout_rate, nb_classes):
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
                  metrics=[get_categorical_accuracy_keras])

    return model


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # load data
    texts, labels = load_data(
        args['DATA_PATH'], bool(args['--verbose']))

    # process data
    data, tokenizer = process_data(
        texts,
        int(args['--words']), int(args['--sent-length']))

    logger.debug('Shape of data tensor: %s', data.shape)
    logger.debug('Shape of label tensor: %s', labels.shape)

    nb_classes = labels.shape[1]

    # make split
    x_train, y_train, x_val, y_val = train_val_split(
        data,
        labels,
        float(args['--test-size']),
        int(args['--min-count']))

    # build and train a model
    logger.info('Building model')
    model = build_model(tokenizer.word_index,
                        args['GLOVE'],
                        int(args['--sent-length']),
                        float(args['--dropout']),
                        nb_classes)

    logger.info('Printing model summary')
    model.summary()
    callbacks = []
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=int(args['--early-stopping-patience'])))

    logger.info('Fit that thing!')
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        epochs=int(args['--epochs']),
        batch_size=int(args['--batch-size']),
        verbose=int(args['--verbose']))

    # evalute model on train data to see how well we're fitting the data
    logger.info("Train data")
    logger.info(model.evaluate(x_train, y_train, batch_size=128))

    # evalute model on validation data
    logger.info("Validation data")
    logger.info(model.evaluate(x_val, y_val, batch_size=128))

    # all new operations will be in test mode from now on (dropout, etc.)
    K.set_learning_phase(0)
    saver = tf.train.Saver()

    with K.get_session() as sess:
        saver.save(sess, os.path.join(args['OUTPUT_PATH'], 'model'))

    with open(os.path.join(args['OUTPUT_PATH'], 'tokenizer.pickle'), 'wb') as f:
        pickle.dump(tokenizer, f, protocol=2)


if __name__ == '__main__':
    args = docopt(__doc__, version='0.1')
    main(args)

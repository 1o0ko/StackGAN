"""
Use a pretrained language model to embed text datasets

Usage: graph_freeze.py BIRDS_DIR MODEL_DIR TOKENIZER_PATH [options]

Arguments:
    BIRD_DIR        directory of birds dataset
    MODEL_PATH      path to model graph
    TOKENIZER_PATH  path to pickled tokenizer

Oprions:
    -r, --reader=<str>  Picks specific data reader
                        [default: birds]
"""
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',)

import numpy as np
import pickle
import tensorflow as tf

from model import Model
from readers import readers
from docopt import docopt

MODEL_NAME = 'custom_embeddings.pickle'


def load_filenames(data_dir):
    logger = logging.getLogger(__name__)
    filepath = os.path.join(data_dir, 'filenames.pickle')

    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)

    logger.info('Load filenames from: %s (%d)' % (filepath, len(filenames)))

    return filenames


def embed_and_save(inpath, filenames, outpath, model, reader):
    logger = logging.getLogger(__name__)

    logger.info('Embedding texts')
    embeddings = [model.embed(texts) for texts in reader(inpath, filenames)]

    embedding_path = os.path.join(outpath, MODEL_NAME)
    logger.info('Saving embeddings to %s' % (embedding_path))
    with open(embedding_path, 'wb') as f:
        pickle.dump(embeddings, f)


def convert_birds_dataset_pickle(data_path, model, reader):
    train_dir = os.path.join(data_path, 'train/')
    train_filenames = load_filenames(train_dir)
    embed_and_save(data_path, train_filenames, train_dir, model, reader)

    test_dir = os.path.join(data_path, 'test/')
    test_filenames = load_filenames(test_dir)
    embed_and_save(data_path, test_filenames, test_dir, model, reader)


if __name__ == '__main__':
    args = docopt(__doc__, version='0.1')
    print(args)

    model = Model(args['MODEL_DIR'], args['TOKENIZER_PATH'])
    reader = readers[args['--reader']]
    convert_birds_dataset_pickle(args['BIRDS_DIR'], model, reader)

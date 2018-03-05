"""
Use a pretrained language model to embed text datasets

Usage: graph_freeze.py DATA_DIR MODEL_DIR TOKENIZER_PATH [options]

Arguments:
    DATA_DIR                    Directory of birds dataset
    MODEL_PATH                  Path to model graph
    TOKENIZER_PATH              Path to pickled tokenizer

Options:
    -r, --reader=<str>          Picks specific data reader
                                [default: birds]

    -m, --embeddings-name=<str> Name used to saved pickled model
                                [default: custom-embeddings]
"""
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',)

import pickle

from model import Model
from readers import readers
from utils import load_filenames

from docopt import docopt


def embed_and_save(inpath, filenames, outpath, model, embeddings_name, reader):
    logger = logging.getLogger(__name__)

    logger.info('Embedding texts')
    embeddings = [model.embed(texts) for texts in reader(inpath, filenames)]

    embedding_path = os.path.join(outpath, embeddings_name)
    logger.info('Saving embeddings to %s' % (embedding_path))
    with open(embedding_path, 'wb') as f:
        pickle.dump(embeddings, f)


def convert_birds_dataset_pickle(data_path, model, embeddings_name, reader):
    train_dir = os.path.join(data_path, 'train/')
    train_filenames = load_filenames(train_dir)
    embed_and_save(data_path, train_filenames, train_dir, model, embeddings_name, reader)

    test_dir = os.path.join(data_path, 'test/')
    test_filenames = load_filenames(test_dir)
    embed_and_save(data_path, test_filenames, test_dir, model, embeddings_name, reader)


if __name__ == '__main__':
    args = docopt(__doc__, version='0.1')

    embeddings_name = "%s.pickle" % args['--embeddings-name']
    model = Model(args['MODEL_DIR'], args['TOKENIZER_PATH'])
    reader = readers[args['--reader']]
    convert_birds_dataset_pickle(args['DATA_DIR'], model, embeddings_name, reader)

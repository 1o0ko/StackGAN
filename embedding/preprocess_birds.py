"""
Usage: preprocess_birds.py BIRDS_DIR [options]

Arguments:
    BIRDS_DIR      path with data

Options:
    -h, --help     display help
"""
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',)

import pickle

from docopt import docopt
from itertools import chain, islice

from readers import readers
from utils import load_filenames

PREFIX='__label__'

def zip_and_save(inpath, filenames, outpath, reader):
    logger = logging.getLogger(__name__)
    with open(os.path.join(outpath, 'class_info.pickle')) as f:
        classes = pickle.load(f)
        classes = ["%s%s" % (PREFIX, cls) for cls in classes]

    texts = [text for text in reader(inpath, filenames)]
    logger.info("Load %i texts" % len(texts))

    for cls, txts in zip(classes, texts):
        for txt in txts:
            yield "%s %s\n" % (cls, txt)


def preprocess_birds(data_path, reader):
    train_dir = os.path.join(data_path, 'train/')
    train_filenames = load_filenames(train_dir)
    train_lines = zip_and_save(data_path, train_filenames, train_dir, reader)

    test_dir = os.path.join(data_path, 'test/')
    test_filenames = load_filenames(test_dir)
    test_lines = zip_and_save(data_path, test_filenames, test_dir, reader)

    with open(os.path.join(data_path, 'birds.txt'), 'wt') as f:
        f.writelines(chain(train_lines, test_lines))

if __name__ == '__main__':
    args = docopt(__doc__, version='0.1')
    preprocess_birds(args['BIRDS_DIR'],  readers['birds'])

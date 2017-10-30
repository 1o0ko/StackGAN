import logging
import os
import pickle


def load_filenames(data_dir):
    logger = logging.getLogger(__name__)
    filepath = os.path.join(data_dir, 'filenames.pickle')

    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)

    logger.info('Load filenames from: %s (%d)' % (filepath, len(filenames)))

    return filenames

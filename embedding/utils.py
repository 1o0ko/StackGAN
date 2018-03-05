import logging
import os
import pickle
import string

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

BLACK_LIST = string.punctuation.replace('%', '') + '\n'


def normalize(text,
              black_list=BLACK_LIST,
              vocab=None, lowercase=True, tokenize=False):
    if black_list:
        text = text.translate(string.maketrans(BLACK_LIST, ' ' * len(BLACK_LIST)))
    if lowercase:
        text = text.lower()
    if vocab:
        text = ' '.join([word for word in text.split() if word in vocab])
    if tokenize:
        return text.split()
    else:
        return ' '.join(text.split())


def load_filenames(data_dir):
    logger = logging.getLogger(__name__)
    filepath = os.path.join(data_dir, 'filenames.pickle')

    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)

    logger.info('Load filenames from: %s (%d)' % (filepath, len(filenames)))

    return filenames


def plot_splits(train_classes, test_classes):
    train_c = Counter(train_classes)
    val_c = Counter(test_classes)
    keys = [key for (key, vlue) in train_c.most_common()]

    values_t = np.array([train_c.get(key, 0) for key in keys])
    values_v = np.array([val_c.get(key, 0) for key in keys])

    width = 1
    indexes = np.arange(len(keys))

    plt.figure(figsize=(15, 5))
    plt.bar(indexes, np.log(values_t), width, alpha=0.5, label='train')
    plt.bar(indexes, np.log(values_v), width, color='red', alpha=0.5, label='eval')
    plt.xticks(indexes + width * 0.5, keys, rotation=90)
    plt.title('Compare the distribution of classes in train and eval datasets')
    plt.legend()
    plt.show()

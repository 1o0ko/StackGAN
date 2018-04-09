import itertools
import json
import logging
import os
import string

from collections import Counter

BLACK_LIST = string.punctuation.replace('%', '') + '\n'
BATCH_SIZE = 256


def normalize_class(class_name):
    return class_name.replace(" ", "_").replace("&", "AND")


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


def save_classes_and_texts(data_set, output_dir,
                           batch_size=BATCH_SIZE, limit=None, vocab=None):
    '''
    Dumps the hdf5 dataset to flat textfile, saves categories and
    '''
    logger = logging.getLogger(__name__)
    N = data_set.num_examples
    num_batch = N / batch_size + 1

    token_counter, class_counter = Counter(), Counter()
    handle = data_set.open()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'classes_and_texts.txt')
    with open(output_file, 'wt') as f:
        processed = 0
        for i in itertools.islice(xrange(num_batch), limit):
            # fetch batch of data
            low, high = i * batch_size, min((i + 1) * batch_size, data_set.num_examples)
            rows = data_set.get_data(handle, slice(low, high))

            # process batch
            classes = [normalize_class(row[0]) for row in rows[1]]
            texts = [normalize(text[0], vocab=vocab) for text in rows[4]]
            lines = ["%s %s\n" % (c, t) for c, t in zip(classes, texts)]

            # dumplines
            f.writelines(lines)

            # track progress
            processed += len(texts)
            class_counter.update(classes)
            token_counter.update(itertools.chain(*[text[0].split() for text in texts]))

            if i and i % 100 == 0:
                percent = int(((100.0 * i) / num_batch))
                logger.info("Low: %d, high: %d" % (low, high))
                logger.info("Processing %i batch out of %i [%i processed | %d]" % (i, num_batch, processed, percent))
                logger.debug("Number of tokens in the dictionary: %i" % len(token_counter))
                logger.debug("Number of classes in the dictionary: %i" % len(class_counter))

    logger.debug('Saving class dictionary')
    output_json = os.path.join(output_dir, 'categories.json')
    with open(output_json, 'wt') as f:
        category2idx = {
            key.replace("&", "AND"): idx for idx, (key, count) in enumerate(class_counter.most_common(), 1)
        }
        f.write(json.dumps(category2idx))

    logger.debug('Saving token counts')
    output_json = os.path.join(output_dir, 'tokens.json')
    with open(output_json, 'wt') as f:
        f.write(json.dumps(token_counter.most_common()))

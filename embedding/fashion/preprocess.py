"""
Usage: example.py SOURCE_TEMPLATE OUTPUT_PATH [options]

Arguments:
    SOURCE_TEMPLATE         glob template of the source dataset(s)
    OUTPUT_PATH             path to dump the processed files

Options:
    -l, --limit=<int>       limit on the number of batches processed
    -b, --batch-size=<int>  Size of the batch used for processing
                            [default: 256]
    -s, --sentences=<int>   Take first n-sentences from description
                            [default: 2]
"""
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',)

import itertools
import glob
import json
import logging

from collections import Counter

from docopt import docopt
from fuel.datasets.hdf5 import H5PYDataset
from tqdm import tqdm

from embedding.preprocessing import normalize, normalize_class


def save_classes_and_texts(data_set, output_dir, batch_size,
                           limit=None, vocab=None, head=None):
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
    logger.debug('Saving classes and texts to: %s' % output_file)
    seen = set()
    with open(output_file, 'wt') as f:
        processed = 0
        for i in tqdm(itertools.islice(xrange(num_batch), limit)):
            # fetch batch of data
            low, high = i * batch_size, min((i + 1) * batch_size, data_set.num_examples)
            rows = data_set.get_data(handle, slice(low, high))

            # process batch
            classes = [normalize_class(row[0]) for row in rows[0]]
            texts = [normalize(text[0], vocab=vocab, head=head) for text in rows[1]]
            lines = ["%s %s\n" % (c, t) for c, t in zip(classes, texts)]

            # dumplines (there is a lot of duplication)
            lines = list(set(lines))
            lines = [line for line in lines if line not in seen]
            f.writelines(lines)

            # track progress
            processed += len(texts)
            class_counter.update(classes)
            token_counter.update(itertools.chain(*[text.split() for text in texts]))
            seen.update(lines)
            if i and i % 100 == 0:
                percent = int(((100.0 * i) / num_batch))
                logger.info("Low: %d, high: %d" % (low, high))
                logger.info("Processing %i batch out of %i [%i processed | %d]" % (i, num_batch, processed, percent))
                logger.debug("Number of tokens in the dictionary: %i" % len(token_counter))
                logger.debug("Number of classes in the dictionary: %i" % len(class_counter))

    output_json = os.path.join(output_dir, 'categories.json')
    logger.debug('Saving class dictionary to: %s' % output_json)
    with open(output_json, 'wt') as f:
        category2idx = {
            key: idx for idx, (key, count) in enumerate(class_counter.most_common(), 1)
        }
        f.write(json.dumps(category2idx))

    output_json = os.path.join(output_dir, 'tokens.json')
    logger.debug('Saving tokens to: %s' % output_json)
    with open(output_json, 'wt') as f:
        f.writelines(["%s %s\n" % (t, c) for t, c in token_counter.most_common()])


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug(args)

    # parse arguments
    batch_size = int(args['--batch-size'])
    limit = int(args['--limit']) if args['--limit'] else None
    sentences = int(args['--sentences']) if args['--sentences'] else None
    for data_path in glob.glob(args['SOURCE_TEMPLATE']):
        data_set = H5PYDataset(data_path, which_sets=('all',))
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        out_dir = os.path.join(args['OUTPUT_PATH'], data_name)
        save_classes_and_texts(data_set, out_dir, batch_size, limit, head=sentences)


if __name__ == '__main__':
    args = docopt(__doc__, version='0.1')
    main(args)

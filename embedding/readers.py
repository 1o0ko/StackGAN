'''
Birds, flowers and fashion will have different readers.
'''

import logging
import os


def bird_reader(inpath, filenames):
    logger = logging.getLogger(__name__)

    file_template = os.path.join(inpath, 'text_c10', "%s.txt")
    for i, filename in enumerate(filenames):

        if i % 100 == 0:
            logger.info("%d is being processed: %s" % (i, filename))

        with open(file_template % filename, 'rt') as f:
            # remove newline characters
            lines = [line[:-1] for line in f.readlines()]

            yield lines


# do this using some python magic
readers = {
    'birds': bird_reader
}

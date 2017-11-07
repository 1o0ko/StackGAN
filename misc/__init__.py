from __future__ import division
from __future__ import print_function

import re
import os

from os.path import dirname, join
from glob import glob

PATTERN = 'class (.+?)\(BaseDataset\):'

def find_classes(module):
    with open(module, 'rt') as f:
        for line in f.readlines():
            m = re.search(PATTERN, line)
            if m:
               yield m.group(1)

# import all classes that subclass from PATTERN
for module in glob(join(dirname(__file__), '*.py')):
    if not module.startswith('__'):
        module_name, _ = os.path.splitext(os.path.basename(module))
        classes = list(find_classes(module))
        if classes:
            __import__(module_name, globals(), locals(), classes)

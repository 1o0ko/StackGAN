import sys
import os
WORKDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(WORKDIR)

from misc.registry import datastore

def load_cfg():
    import argparse
    import yaml

    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser(description='Test dataset factory')
    parser.add_argument('--path', dest = 'data_path',
                        default='/data/', type=str)
    parser.add_argument('--cfg', dest='cfg', type=str)

    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = edict(yaml.load(f))

    return args, cfg

if __name__ == '__main__':
    args, cfg = load_cfg()
    datadir = '%s%s' % (args.data_path, cfg.DATASET_NAME)
    dataset = datastore.create(datadir, cfg)

    print("Datastore: %s" % datastore)
    print(dataset)

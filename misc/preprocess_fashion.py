'''
Custom script to preprocess fashion dataset
'''
import os

FASHION_DIR = '/data/fashion/'


def load_filenames(data_dir):
    pass


def save_data_list(inpath, outpath, filenames):
    pass


def convert_fashion_dataset_pickle(inpath):
    # ## For Train data
    train_dir = os.path.join(inpath, 'train/')
    train_filenames = load_filenames(train_dir)
    save_data_list(inpath, train_dir, train_filenames)

    # ## For Test data
    test_dir = os.path.join(inpath, 'test/')
    test_filenames = load_filenames(test_dir)
    save_data_list(inpath, test_dir, test_filenames)

if __name__ == '__main__':
    convert_fashion_dataset_pickle(FASHION_DIR)

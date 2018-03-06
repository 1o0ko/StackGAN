"""
Uses pretrained embedding model to embed descriptions from HDF5 file
and save the datasource format siutable for StackGAN
"""
import json
import os
import pickle

import numpy as np

from fuel.datasets.hdf5 import H5PYDataset
from scipy.misc import imresize

from model import Model
from train_embedding_model import get_split
from utils import normalize

# Pipeline properties
BATH_SIZE = 100
LIMIT = 70

# Img resizing stuff
IMG_SIZE = 256
LR_HR_RATIO = 4
BIG_SIZE = int(IMG_SIZE * 76 / 64)
SMALL_SIZE = int(BIG_SIZE / LR_HR_RATIO)

# Paths
FASHION_PATH = '/fashion/'
DATA_PATH = '/data/fashion/'
MODEL_PATH = '/models/fashion/'
DATA_TEMPLATE = os.path.join(FASHION_PATH, 'ssense_%i_%i.h5')


def create_captions(
        classes, texts, category2idx,
        verbose=True, save=True):
    '''
    helper function to create text_c10 folder
    '''
    cls2count = {k.replace(" ", "_"): 1 for k in category2idx}
    filenames = []
    for index, (cls, text) in enumerate(zip(classes, texts)):
        category = cls.replace(" ", "_").replace("&", 'AND')
        cls = cls.replace("&", 'AND')

        dirname = "%.3i.%s" % (category2idx[cls], category)
        filename = "%s_%i.txt" % (category, cls2count[category])

        directory = os.path.join(DATA_PATH, "text_c10/%s" % dirname)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if verbose and (index % 5000) == 0:
            print("%i - %s" % (index, filename))

        if save:
            with open(os.path.join(directory, filename), 'wt') as f:
                f.write("%s\n" % normalize(text))

        filenames.append(os.path.join(dirname, filename))
        cls2count[category] += 1

    return filenames


def get_batch(list_, batch_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(list_), batch_size):
        yield list_[i:i + batch_size]


def prepare_embeddings(texts, model, limit=None, batch_size=128):
    # normalize texts
    texts_ = [normalize(text) for text in texts[:limit]]

    hs = []
    for index, batch in enumerate(get_batch(texts_, batch_size)):
        if index and index % 100 == 0:
            print("Processing batch number %i" % index)

        hs.extend([h.reshape(1, -1) for h in model.embed(batch)])

    return hs


def dump_all(class_info, filenames, images, texts, split, model, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print('Selecting splits')
    imgs = images[split]
    txts = texts[split]

    class_info_ = np.array(class_info)[split].tolist()
    filenames_ = np.array(filenames)[split].tolist()

    print("N. files: %i, fst: %s" % (len(filenames_), filenames_[0]))
    print("N. examples: %i, fst: %s" % (len(class_info_), class_info_[0]))

    print("Saving class info")
    with open(os.path.join(outdir, 'class_info.pickle '), 'wb') as f:
        pickle.dump(class_info_, f)

    print("Saving filenames")
    with open(os.path.join(outdir, 'filenames.pickle '), 'wb') as f:
        pickle.dump(filenames_, f)

    print('Creating 76x76 images')
    img_76 = [imresize(img, [SMALL_SIZE, SMALL_SIZE], 'bicubic') for img in imgs]
    print("Small images: %i, %s" % (len(img_76), img_76[0].shape))

    print("Saving 76x76 images")
    with open(os.path.join(outdir, '76images.pickle'), 'wb') as f:
        pickle.dump(img_76, f)

    print('Creating 304x304 images')
    img_304 = [imresize(img, [BIG_SIZE, BIG_SIZE], 'bicubic') for img in imgs]
    print("Big images: %i, %s" % (len(img_304), img_304[0].shape))

    print("Saving 304x304 images")
    with open(os.path.join(outdir, '304images.pickle'), 'wb') as f:
        pickle.dump(img_304, f)

    print("Creating text embeddings")
    embeddings = prepare_embeddings(txts, model)
    print("Embeddings %i, %s" % (len(embeddings), embeddings[0].shape))

    print("Saving embeddings")
    with open(os.path.join(outdir, 'custom_embeddings.pickle'), 'wb') as f:
        pickle.dump(embeddings, f)


def main():
    print('Loading categories')
    category2idx = json.load(open(os.path.join(DATA_PATH, 'categories.json'), 'rt'))

    print('Loading data in memory')
    dataset = H5PYDataset(DATA_TEMPLATE % (IMG_SIZE, IMG_SIZE),
                          sources=['input_category', 'input_description', 'input_image'],
                          which_sets=('all',), load_in_memory=True)

    classes, texts, images = dataset.data_sources
    classes = np.array([cls[0] for cls in classes])
    texts = np.array([txt[0] for txt in texts])

    print("There are %i examples" % dataset.num_examples)
    print(texts.shape, images.shape, classes.shape)
    print("N. examples: %i, fst: %s" % (len(classes), classes[0]))

    # prepare filenames
    print("Creating filenames")
    filenames = create_captions(classes, texts, category2idx, False, False)
    print("N. files: %i, fst: %s" % (len(filenames), filenames[0]))

    train_idx, test_idx, _, _ = get_split(classes, classes.reshape(-1, 1), 0.1, seed=2)

    print('Loading embedding model')
    model = Model(
        os.path.join(MODEL_PATH, 'frozen_model.pb'),
        os.path.join(MODEL_PATH, 'tokenizer.pickle'),
        maxlen=LIMIT
    )

    print('Saving test data')
    dump_all(classes, filenames, images, texts, test_idx, model,
            os.path.join(DATA_PATH, 'test'))

    print('Saving train data')
    dump_all(classes, filenames, images, texts, train_idx, model,
            os.path.join(DATA_PATH, 'train'))


if __name__ == '__main__':
    main()

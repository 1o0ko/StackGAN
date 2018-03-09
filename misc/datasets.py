from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import random

from registry import register, datastore


'''
 Stuff that is calling the dataset object uses this methods
 dataset.embedding_shape,
 dataset.image_shape,
 dataset._num_examples:
 dataset.hr_lr_ratio
 dataset.next_batch_test(self.batch_size, start, 1)


 dataset exposes test/train with
  self.next_batch(n * n, 1)
  self.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
  self._num_examples
'''


class BaseDataset(object):
    def __init__(self, workdir, embedding_filename, hr_lr_ratio):
        self.workdir = workdir
        self.embedding_filename = embedding_filename
        self.hr_lr_ratio = hr_lr_ratio

    def __repr__(self):
        return "[%s]\nWorkdir: %s\nEmbedding filename: %s\nHR to LR ration: %i" % (
            self.__class__.__name__, self.workdir, self.embedding_filename, self.hr_lr_ratio)


class Dataset(object):
    def __init__(self, images, imsize, embeddings=None,
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None):
        self._images = images
        self._embeddings = embeddings
        self._filenames = filenames
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(images)
        self._saveIDs = self.saveIDs()

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._aug_flag = aug_flag
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None

    @property
    def images(self):
        return self._images

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def saveIDs(self):
        self._saveIDs = np.arange(self._num_examples)
        np.random.shuffle(self._saveIDs)
        return self._saveIDs

    def readCaptions(self, name, class_id):
        if name.find('jpg/') != -1:  # flowers dataset
            class_name = 'class_%05d/' % class_id
            name = name.replace('jpg/', class_name)

        # TODO: this is fugly, change the fashion stuff 
        if 'fashion' in self.workdir:
            cap_path = '%s/text_c10/%s' % (self.workdir, name)
        else:
            cap_path = '%s/text_c10/%s.txt' % (self.workdir, name)

        with open(cap_path, "r") as f:
            captions = f.read().split('\n')

        captions = [cap for cap in captions if len(cap) > 0]

        return captions

    def transform(self, images):
        if self._aug_flag:
            transformed_images =\
                np.zeros([images.shape[0], self._imsize, self._imsize, 3])
            ori_size = images.shape[1]
            for i in range(images.shape[0]):
                h1 = int(np.floor((ori_size - self._imsize) * np.random.random()))
                w1 = int(np.floor((ori_size - self._imsize) * np.random.random()))
                cropped_image =\
                    images[i][w1: w1 + self._imsize, h1: h1 + self._imsize, :]
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(cropped_image)
                else:
                    transformed_images[i] = cropped_image
            return transformed_images
        else:
            return images

    def sample_embeddings(self, embeddings, filenames, class_id, sample_num):
        batch_size, embedding_num, _ = embeddings.shape
        sampled_captions = []

        if embedding_num == 1 or len(embeddings.shape) == 2:
            for i in range(batch_size):
                captions = self.readCaptions(filenames[i], class_id[i])
                sampled_captions.append(captions[0])
            return np.squeeze(embeddings), sampled_captions
        else:
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = []
            for i in range(batch_size):
                randix = np.random.choice(embedding_num,
                                          sample_num, replace=False)
                if sample_num == 1:
                    randix = int(randix)
                    captions = self.readCaptions(filenames[i], class_id[i])
                    sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    # I don't know why, but we're not sampling captions here
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def next_batch(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        current_ids = self._perm[start:end]
        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        collision_flag =\
            (self._class_id[current_ids] == self._class_id[fake_ids])
        fake_ids[collision_flag] =\
            (fake_ids[collision_flag] +
             np.random.randint(100, 200)) % self._num_examples

        sampled_images = self._images[current_ids]
        sampled_wrong_images = self._images[fake_ids, :, :, :]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.

        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [sampled_images, sampled_wrong_images]

        if self._embeddings is not None:
            filenames = [self._filenames[i] for i in current_ids]
            class_id = [self._class_id[i] for i in current_ids]
            try:
                sampled_embeddings, sampled_captions = \
                    self.sample_embeddings(
                        self._embeddings[current_ids],
                        filenames, class_id, window)
            except Exception:
                import ipdb; ipdb.set_trace()
                raise Exception

            ret_list.append(sampled_embeddings)
            ret_list.append(sampled_captions)
        else:
            ret_list.append(None)
            ret_list.append(None)

        if self._labels is not None:
            ret_list.append(self._labels[current_ids])
        else:
            ret_list.append(None)
        return ret_list

    def next_batch_test(self, batch_size, start, max_captions):
        """Return the next `batch_size` examples from this data set."""
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            start = end - batch_size
        else:
            end = start + batch_size

        sampled_images = self._images[start:end]
        sampled_images = sampled_images.astype(np.float32)
        # from [0, 255] to [-1.0, 1.0]
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_images = self.transform(sampled_images)

        sampled_embeddings = self._embeddings[start:end]
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []

        sampled_captions = []
        sampled_filenames = self._filenames[start:end]
        sampled_class_id = self._class_id[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i],
                                         sampled_class_id[i])
            # print(captions)
            sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(np.squeeze(batch))

        return [sampled_images, sampled_embeddings_batchs,
                self._saveIDs[start:end], sampled_captions]


@register(default=True)
class TextDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(TextDataset, self).__init__(*args, **kwargs)

        self.lr_imsize = 64

        if self.hr_lr_ratio == 1:
            self.image_filename = '/76images.pickle'
        elif self.hr_lr_ratio == 4:
            self.image_filename = '/304images.pickle'

        self.image_shape = [self.lr_imsize * self.hr_lr_ratio,
                            self.lr_imsize * self.hr_lr_ratio, 3]
        self.image_dim = self.image_shape[0] * self.image_shape[1] * 3
        self.embedding_shape = None
        self.train = None
        self.test = None

    def get_data(self, pickle_path, aug_flag=True):
        print('Pickle path: %s' % pickle_path)

        embeddings_path = os.path.join(pickle_path, self.embedding_filename)
        print('Embeddings path: %s' % embeddings_path)
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            self.embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)

        filenames_path = os.path.join(pickle_path, '/filenames.pickle')
        print('filenames path: %s' % filenames_path)
        with open(filenames_path, 'rb') as f:
            list_filenames = pickle.load(f)
            print('list_filenames: ', len(list_filenames), list_filenames[0])

        class_info_path = os.path.join(pickle_path, '/class_info.pickle')
        print('class_info_path: %s' % class_info_path)
        with open(class_info_path, 'rb') as f:
            class_id = pickle.load(f)
            print('class_id: ', len(class_id), class_id[0])

        images_path = os.path.join(pickle_path, self.image_filename)
        print('images path: %s' % images_path)
        with open(images_path, 'rb') as f:
            images = pickle.load(f)
            images = np.array(images)
            print('images: ', images.shape)

        return Dataset(images, self.image_shape[0], embeddings,
                       list_filenames, self.workdir, None,
                       aug_flag, class_id)


if __name__ == '__main__':
    import argparse
    import yaml

    from easydict import EasyDict as edict
    parser = argparse.ArgumentParser(description='Test dataset factory')
    parser.add_argument('--path', dest='data_path', default='/data/', type=str)
    parser.add_argument('--cfg', dest='cfg', type=str)

    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = edict(yaml.load(f))

    datadir = '%s%s' % (args.data_path, cfg.DATASET_NAME)
    dataset = datastore.create(datadir, cfg)

    print("Datastore: %s" % datastore)
    print(dataset)

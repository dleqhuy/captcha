import os
import re

import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers

try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError:
    # tf < 2.4.0
    AUTOTUNE = tf.data.experimental.AUTOTUNE

class DatasetBuilder:

    def __init__(self, table_path, img_shape=(60, 200, 3)):
        with open(table_path, 'r') as f:
            vocab = [line.rstrip('\n') for line in f]
        self.char_to_num = layers.StringLookup(vocabulary=vocab, mask_token=None, output_mode='multi_hot', sparse=True)
        self.num_to_char = layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)

        self.img_shape = img_shape

    @property
    def num_classes(self):
        return len(self.char_to_num.get_vocabulary())

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_shape[-1])
        img = tf.image.resize(img, (self.img_shape[1], self.img_shape[0])) / 255.0
        img = tf.transpose(img, perm=[1, 0, 2])
        return img, label

    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = self.char_to_num(chars)

        return imgs, tokens

    def __call__(self, dataframe, batch_size, is_training):

        ds = tf.data.Dataset.from_tensor_slices((dataframe['file_path'], dataframe['label']))

        if is_training:
            ds = ds.shuffle(buffer_size=1000)

        ds = ds.map(self._decode_img, AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=is_training)
        ds = ds.map(self._tokenize, AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

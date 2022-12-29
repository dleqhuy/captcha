import os
import re

import tensorflow as tf
import pandas as pd
try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError:
    # tf < 2.4.0
    AUTOTUNE = tf.data.experimental.AUTOTUNE

class DatasetBuilder:

    def __init__(self, table_path, img_shape=(60, 200, 3)):
        # map unknown label to 0
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), 0)
        self.img_shape = img_shape

    @property
    def num_classes(self):
        return self.table.size()

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_shape[-1])
        img = tf.image.resize(img, (self.img_shape[1], self.img_shape[0])) / 255.0
        img = tf.transpose(img, perm=[1, 0, 2])
        return img, label

    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        # TODO(hym) Waiting for official support to use RaggedTensor in keras
        tokens = tokens.to_sparse()
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

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

    def __init__(self, table_path, img_width, img_height, channel, is_handwriting):
        
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
                    table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
                    tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), 0)
        
        self.img_width = img_width
        self.img_height = img_height
        self.channel = channel
        self.is_handwriting = is_handwriting
    @property
    def num_classes(self):
        return self.table.size()

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_png(img, channels=self.channel)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label
    
    def _resize(self, img, label):
        
        img = tf.image.resize(img, (self.img_height, self.img_width))
        img = tf.transpose(img, perm=[1, 0, 2])
        return img, label
    
    def _distortion_free_resize(self, img, label):
        img = tf.image.resize(img, size=(self.img_height, self.img_width), preserve_aspect_ratio=True)

        # Check tha amount of padding needed to be done.
        pad_height = self.img_height - tf.shape(img)[0]
        pad_width = self.img_width - tf.shape(img)[1]

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        img = tf.pad(
            img,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )

        img = tf.transpose(img, perm=[1, 0, 2])
        img = tf.image.flip_left_right(img)
        return img, label
    
    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        # TODO(hym) Waiting for official support to use RaggedTensor in keras
        tokens = tokens.to_sparse()
        return imgs, tokens

    def __call__(self, dataframe, batch_size, cache=False, shuffle=False, drop_remainder=False):

        ds = tf.data.Dataset.from_tensor_slices((dataframe['file_path'], dataframe['label']))
        ds = ds.map(self._decode_img, AUTOTUNE)
        
        if shuffle:
            ds = ds.shuffle(buffer_size=500)
        if self.is_handwriting:
            ds = ds.map(self._distortion_free_resize, AUTOTUNE)
        else:
            ds = ds.map(self._resize, AUTOTUNE)

        if cache:
            ds = ds.cache()
            
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.map(self._tokenize, AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

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

    def __init__(self, vocab_path, img_width, img_height, channel):
        with open(vocab_path) as file:
            vocab = [line.rstrip() for line in file]
        self.char_to_num = layers.StringLookup(
            vocabulary=vocab, mask_token=None,
        )

        self.img_width = img_width
        self.img_height = img_height
        self.channel = channel
    @property
    def num_classes(self):
        return len(self.char_to_num.get_vocabulary())+1

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_png(img, channels=self.channel)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (self.img_height, self.img_width))
        img = tf.transpose(img, perm=[1, 0, 2])
                
        return img, label
    
    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.char_to_num, chars)
        # TODO(hym) Waiting for official support to use RaggedTensor in keras
        tokens = tokens.to_sparse()
        return imgs, tokens
    
    def __call__(self, dataframe, batch_size, cache=False, shuffle=False, drop_remainder=False):

        ds = tf.data.Dataset.from_tensor_slices((dataframe['file_path'], dataframe['label']))
        
        if shuffle:
            ds = ds.shuffle(buffer_size=500)
            
        ds = ds.map(self._decode_img, AUTOTUNE)
        
        if cache:
            ds = ds.cache()
            
        ds = ds.padded_batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.map(self._tokenize, AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        return ds
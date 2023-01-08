import argparse
from pathlib import Path

import glob
import yaml
from tensorflow import keras
import tensorflow as tf

from models import build_model
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, 
                    help='The config file path.')
parser.add_argument('--weight', type=str, required=True, default='',
                    help='The saved weight path.')
parser.add_argument('--post', type=str, help='Post processing.')
parser.add_argument('--images', type=str, required=True, 
                    help='Image file or folder path.')

args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']


with open(config['table_path']) as f:
    num_classes = len(f.readlines())
    
if args.post == 'greedy':
    postprocess = CTCGreedyDecoder(config['table_path'])
elif args.post == 'beam_search':
    postprocess = CTCBeamSearchDecoder(config['table_path'])
else:
    postprocess = None

model = build_model(num_classes, 
                    weight=args.weight,
                    img_width=config['img_width'],
                    img_height=config['img_height'],
                    channel=config['channel']
                   )

def distortion_free_resize(img,img_height,img_width):
        img = tf.image.resize(img, size=(img_height, img_width), preserve_aspect_ratio=True)

        # Check tha amount of padding needed to be done.
        pad_height = img_height - tf.shape(img)[0]
        pad_width = img_width - tf.shape(img)[1]

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
        return img
    

def read_img_and_resize(path, img_width, img_height, channel):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=channel)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if config['is_handwriting']:
        img = distortion_free_resize(img,img_height, img_width)
    else:
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.transpose(img, perm=[1, 0, 2])

    return img

img_paths = glob.glob(f'{args.images}/*.png')

for img_path in img_paths:
    img = read_img_and_resize(str(img_path), config['img_width'],  config['img_height'], config['channel'])
    img = tf.expand_dims(img, 0)
    outputs = model(img)
    
    y_pred, probability = postprocess.call(outputs)
    
    print(f'Path: {img_path}, y_pred: {y_pred.numpy().astype(str)}, '
          f'probability: {probability.numpy()}')
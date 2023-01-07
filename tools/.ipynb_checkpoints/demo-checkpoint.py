import argparse
from pathlib import Path

import yaml
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, required=True, 
                    help='Image file or folder path.')
parser.add_argument('--config', type=Path, required=True, 
                    help='The config file path.')
parser.add_argument('--model', type=str, required=True, 
                    help='The saved model.')
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']


def read_img_and_resize(path, shape):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=shape[2])
    img = tf.image.resize(img, (shape[1], shape[0])) / 255.0
    img = tf.transpose(img, perm=[1, 0, 2])

    return img


model = keras.models.load_model(args.model, compile=False)

p = Path(args.images)
img_paths = p.iterdir() if p.is_dir() else [p]
for img_path in img_paths:
    img = read_img_and_resize(str(img_path), config['img_shape'])
    img = tf.expand_dims(img, 0)
    outputs = model(img)
    print(f'Path: {img_path}, y_pred: {outputs[0].numpy()}, '
          f'probability: {outputs[1].numpy()}')
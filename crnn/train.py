import argparse
import pprint
import shutil
from pathlib import Path

import tensorflow as tf
import pandas as pd
import yaml
from tensorflow import keras
from tensorflow.keras import mixed_precision

from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy
from models import build_model
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True,
                    help='The config file path.')
parser.add_argument('--save_dir', type=Path, required=True,
                    help='The path to save the models, logs, etc.')
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
pprint.pprint(config)

args.save_dir.mkdir(exist_ok=True)
# if list(args.save_dir.iterdir()):
#     raise ValueError(f'{args.save_dir} is not a empty folder')
shutil.copy(args.config, args.save_dir / args.config.name)

batch_size = config['batch_size_per_replica']

mixed_precision.set_global_policy('mixed_float16')

dataset_builder = DatasetBuilder(**config['dataset_builder'])

df_sample = pd.read_csv(config['train_csv_path'])
df_sample = df_sample.astype(str)
#added some parameters
kf = KFold(n_splits = 10, shuffle = True, random_state = 2)

for i, (train_index, val_index) in enumerate(kf.split(df_sample)):

    print(f'=================Fold {i}=================')
    train_df = df_sample.iloc[train_index]
    val_df = df_sample.iloc[val_index]

    train_ds = dataset_builder(train_df, batch_size, True)
    val_ds = dataset_builder(val_df, batch_size, False)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        **config['lr_schedule'])
    model = build_model(dataset_builder.num_classes,
                        weight=config.get('weight'),
                        img_shape=config['dataset_builder']['img_shape'])
    model.compile(optimizer=keras.optimizers.Adam(lr_schedule),
                    loss=CTCLoss(), metrics=[SequenceAccuracy()])

    model_prefix = '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'
    model_path = f'{args.save_dir}/{model_prefix}.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path,
                                        save_weights_only=True),
        keras.callbacks.TensorBoard(log_dir=f'{args.save_dir}/logs{i}',
                                    **config['tensorboard'])
    ]


    model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks,
            validation_data=val_ds)

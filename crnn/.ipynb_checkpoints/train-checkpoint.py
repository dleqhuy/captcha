import argparse
import pprint
import shutil
from pathlib import Path

import tensorflow as tf
import pandas as pd
import yaml
from tensorflow import keras

from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy
from models import build_model
from sklearn.model_selection import KFold
from IPython.display import display

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

shutil.copy(args.config, args.save_dir / args.config.name)

batch_size = config['batch_size_per_replica']

dataset_builder = DatasetBuilder(**config['dataset_builder'])

model = build_model(dataset_builder.num_classes,
                    weight=config.get('weight'),
                    img_width=config['dataset_builder']['img_width'],
                    img_height=config['dataset_builder']['img_height'],
                    channel=config['dataset_builder']['channel']
                   )
model.summary()

df_sample = pd.read_csv(config['train_csv_path'])
df_sample = df_sample.astype(str)

df_results = pd.DataFrame(columns=['loss', 'sequence_accuracy', 'val_loss', 'val_sequence_accuracy','epoch'])
df_results.index.name = 'Fold'
#added some parameters
kf = KFold(n_splits = config['num_kfold'],shuffle=True, random_state = 2)

for i, (train_index, val_index) in enumerate(kf.split(df_sample)):

    train_df = df_sample.iloc[train_index]
    val_df = df_sample.iloc[val_index]

    train_ds = dataset_builder(train_df, batch_size, shuffle=True)
    val_ds = dataset_builder(val_df, batch_size)

    model = build_model(dataset_builder.num_classes,
                        weight=config.get('weight'),
                        img_width=config['dataset_builder']['img_width'],
                        img_height=config['dataset_builder']['img_height'],
                        channel=config['dataset_builder']['channel']
                       )
    
    model.compile(optimizer=keras.optimizers.Adam(),
                    loss=CTCLoss(), metrics=[SequenceAccuracy()])

    
    
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=f'{args.save_dir}/logs{i}',
                                    **config['tensorboard']),
        keras.callbacks.EarlyStopping(monitor="val_loss",**config['earlystopping']),

    ]

    history = model.fit(train_ds, epochs=config['epochs'],
                        callbacks=callbacks,
                        verbose=0,
                        validation_data=val_ds)
    
    model_path = f'{args.save_dir}/{i}_best_model.h5'
    model.save(model_path)
    
    df_results.loc[i,'loss'] = history.history['loss'][-1]
    df_results.loc[i,'sequence_accuracy'] = history.history['sequence_accuracy'][-1]
    df_results.loc[i,'val_loss'] = history.history['val_loss'][-1]
    df_results.loc[i,'val_sequence_accuracy'] = history.history['val_sequence_accuracy'][-1]
    df_results.loc[i,'epoch'] = len(history.history['loss'])

df_results.loc['Mean'] = df_results.mean()
df_results.loc['Std'] = df_results.std()

display(df_results)

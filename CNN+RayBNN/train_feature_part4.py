import os
import time

import keras

from keras.layers import concatenate

from sklearn.metrics import cohen_kappa_score

import math
import random
from keras import optimizers
import numpy as np
import tensorflow as tf
import scipy.io as spio
from sklearn.metrics import f1_score, accuracy_score

np.random.seed(0)

from keras.preprocessing import sequence
from keras import utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import GRU, Bidirectional
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, \
    GlobalAveragePooling2D
from keras.callbacks import History
from keras.models import Model, load_model

from collections import Counter

from sklearn.utils.class_weight import compute_class_weight

from myModel import build_model

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from loadData import *
from utils import *

_script_start = time.time()

batch_size = 200
n_ep = 2
fs = 200
w_len = 8 * fs
data_dim = w_len * 2
half_prec = 0.5
prec = 1
n_cl = 4

data_dir = os.path.join(HERE, 'mwt_eeg') + os.sep
f_set = os.path.join(HERE, 'file_sets_part4.mat')

output_dir = os.path.join(HERE, 'output')
models_dir = os.path.join(output_dir, 'models')
predictions_dir = os.path.join(output_dir, 'predictions')

create_tmp_dirs([models_dir, predictions_dir])

mat = spio.loadmat(f_set)

files_train = []
files_val = []
files_test = []

tmp = mat['files_train']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_train.extend(file)

tmp = mat['files_val']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_val.extend(file)
tmp = mat['files_test']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_test.extend(file)


def my_generator(data_train, targets_train, sample_list, shuffle=True):
    if shuffle:
        random.shuffle(sample_list)

    while True:
        for batch in batch_generator(sample_list, batch_size):
            batch_data1 = []
            batch_targets = []
            for sample in batch:
                [f, s, b, e, c] = sample
                sample_label = targets_train[f][c][s]
                sample_x1 = data_train[f][c][b:e + 1]
                sample_x2 = data_train[f][1][b:e + 1]
                sample_x = np.concatenate((sample_x1, sample_x2), axis=2)
                batch_data1.append(sample_x)
                batch_targets.append(sample_label)
            batch_data1 = np.stack(batch_data1, axis=0)
            batch_targets = np.array(batch_targets)
            batch_targets = to_categorical(batch_targets, n_cl)
            batch_data1 = (batch_data1) / 100
            batch_data1 = np.clip(batch_data1, -1, 1)
            yield batch_data1.astype(np.float32), batch_targets.astype(np.float32)


def make_dataset(data, targets, sample_list, shuffle=True):
    def gen():
        yield from my_generator(data, targets, sample_list, shuffle=shuffle)
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, data_dim, 1, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, n_cl), dtype=tf.float32),
        )
    )


n_channels = 2

st0 = classes_global(data_dir, files_train)
cls = np.arange(n_cl)
cl_w = compute_class_weight(class_weight='balanced', classes=cls, y=st0)

(data_train, targets_train, N_samples) = load_data(data_dir, files_train, w_len)

N_batches = int(math.ceil((N_samples + 0.0) / batch_size))

(data_val, targets_val, N_samples_val) = load_data(data_dir, files_val, w_len)

sample_list = []
for ch in range(2):
    for i in range(len(targets_train)):
        for j in range(len(targets_train[i][0])):
            mid = j * prec
            mid += w_len
            wnd_begin = mid - w_len
            wnd_end = mid + w_len - 1
            sample_list.append([i, j, wnd_begin, wnd_end, ch])

sample_list_val = []

for i in range(len(targets_val)):
    sample_list_val.append([])
    for j in range(len(targets_val[i][0])):
        mid = j * prec
        mid += w_len
        wnd_begin = mid - w_len
        wnd_end = mid + w_len - 1
        sample_list_val[i].append([i, j, wnd_begin, wnd_end, 0])

ordering = 'channels_last'
keras.backend.set_image_data_format(ordering)

[cnn_eeg, model] = build_model(data_dim, n_channels, n_cl)
Nadam = optimizers.Nadam()
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Start extracting features and labels...")
print("Loading Model......")
cnn_eeg = load_model(os.path.join(predictions_dir, 'cnn_eeg_model2.keras'))

ds_train = make_dataset(data_train, targets_train, sample_list, shuffle=False)
all_x, all_y = zip(*ds_train.take(N_batches))
features = cnn_eeg.predict(np.concatenate(all_x, axis=0))
labels = np.concatenate(all_y, axis=0)

print("Feature extraction finished.")
print("feature.shape:", features.shape)
print("label.shape:", labels.shape)

N_batches_val = int(math.ceil(N_samples_val / batch_size))
ds_val = make_dataset(data_val, targets_val, sample_list_val[0], shuffle=False)
all_x_val, all_y_val = zip(*ds_val.take(N_batches_val))
features_val = cnn_eeg.predict(np.concatenate(all_x_val, axis=0))
labels_val = np.concatenate(all_y_val, axis=0)

print("feature_val.shape:", features_val.shape)
print("label_val.shape:", labels_val.shape)

np.save(os.path.join(predictions_dir, 'features_part4_0.npy'), features)
np.save(os.path.join(predictions_dir, 'labels_part4_0.npy'), labels)
np.save(os.path.join(predictions_dir, 'features_val_part4_0.npy'), features_val)
np.save(os.path.join(predictions_dir, 'labels_val_part4_0.npy'), labels_val)

_elapsed = time.time() - _script_start
print(f"[train_feature_part4.py] Total runtime: {_elapsed:.1f}s ({_elapsed/3600:.2f}h)")

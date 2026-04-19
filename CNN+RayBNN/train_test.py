import os
import time

import keras

from keras.layers import concatenate

from sklearn.metrics import cohen_kappa_score

import math
import random
from keras import optimizers
import numpy as np

# FIX: Disable XLA and eager execution to prevent generator threading issues
import tensorflow as tf
tf.config.run_functions_eagerly(True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

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
from keras.models import Model

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
f_set = os.path.join(HERE, 'file_sets.mat')

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

history = History()

K_cv_tmp = []
K_tst_tmp = []
K_val = np.zeros((n_ep, n_cl))
K_tst = np.zeros((n_ep, n_cl))

K_tr = []
N_steps = 1000

for i in range(n_ep):
    print("Epoch = " + str(i))
    ds_train = make_dataset(data_train, targets_train, sample_list, shuffle=True)
    model.fit(ds_train, steps_per_epoch=N_batches, epochs=1, verbose=1, callbacks=[history])

    val_y_ = []
    val_y = []
    f_list = files_val
    val_l = []
    for j in range(len(data_val)):
        n_val_steps = int(math.ceil((len(sample_list_val[j]) + 0.0) / batch_size))
        ds_val = make_dataset(data_val, targets_val, sample_list_val[j], shuffle=False)
        scores = model.evaluate(ds_val, steps=n_val_steps)
        ds_val = make_dataset(data_val, targets_val, sample_list_val[j], shuffle=False)
        y_pred = model.predict(ds_val, steps=n_val_steps)
        val_l.append(len(sample_list_val[j]))
        print(len(sample_list_val[j]))
        print(len(targets_val[j][0]))
        val_y_.extend(np.argmax(y_pred, axis=1).flatten())
        val_y.extend(targets_val[j][0])

    t1 = kappa_metric(val_y, val_y_, n_cl)
    K_val_tmp = t1
    K_val[i, :] = K_val_tmp
    t2 = cohen_kappa_score(val_y, val_y_)
    print("K val per class = ", t1)
    print("K val = ", t2)

    model.save(os.path.join(models_dir, 'model_ep2' + str(i) + '.keras'))
    spio.savemat(os.path.join(predictions_dir, 'predictions_ep2' + str(i) + '.mat'),
                 mdict={'val_y': val_y, 'val_y_': val_y_, 'val_l': val_l, 'files_test': files_test,
                        'files_val': files_val, 'K_val': K_val})

model.save(os.path.join(models_dir, 'model2.keras'))

spio.savemat(os.path.join(output_dir, 'predictions2.mat'),
             mdict={'val_y': val_y, 'val_y_': val_y_, 'val_l': val_l, 'files_test': files_test,
                    'files_val': files_val, 'K_val': K_val})

cnn_eeg.save(os.path.join(predictions_dir, 'cnn_eeg_model2.keras'))
model.save(os.path.join(predictions_dir, 'full_model2.keras'))

_elapsed = time.time() - _script_start
print(f"[train_test.py] Total runtime: {_elapsed:.1f}s ({_elapsed/3600:.2f}h)")

import scipy.io as spio
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import math
from random import randint
from keras.utils import to_categorical


def load_recording(dir_name, f_name, n_cl=4):
    mat = spio.loadmat(dir_name + f_name, struct_as_record=False, squeeze_me=True)
    labels_O1 = mat['Data'].labels_O1
    labels_O2 = mat['Data'].labels_O2
    eeg_O1 = mat['Data'].eeg_O1
    eeg_O2 = mat['Data'].eeg_O2

    eeg_O1 = np.expand_dims(eeg_O1, axis=-1)
    eeg_O1 = np.expand_dims(eeg_O1, axis=-1)
    eeg_O2 = np.expand_dims(eeg_O2, axis=-1)
    eeg_O2 = np.expand_dims(eeg_O2, axis=-1)

    targets_O1 = labels_O1
    targets_O2 = labels_O2
    return (eeg_O1, eeg_O2, targets_O1, targets_O2)


def classes(dir_name, f_name):
    mat = spio.loadmat(dir_name + f_name, struct_as_record=False, squeeze_me=True)
    labels_O1 = mat['Data'].labels_O1
    return labels_O1


def classes_global(data_dir, files_train):
    print("=====================")
    print("Reading train set:")
    f_list = files_train
    st0 = []
    for i in range(0, len(f_list)):
        f = f_list[i]
        st = classes(data_dir, f)
        st0.extend(st)
    return st0


def load_data(data_dir, files_train, w_len=200 * 2):
    E1, E2, targets1, targets2 = load_recording(data_dir, files_train[0])
    print(E1.shape)
    data_train = []
    targets_train = []

    f_list = files_train

    padding = ((w_len, w_len), (0, 0), (0, 0))

    N_samples = 0
    for i in range(0, len(f_list)):
        f = f_list[i]
        print(f)
        E1, E2, targets1, targets2 = load_recording(data_dir, f)

        E1 = np.pad(E1, pad_width=padding, mode='constant', constant_values=0)
        E2 = np.pad(E2, pad_width=padding, mode='constant', constant_values=0)
        N_samples += 2 * len(targets1)

        data_train.append([E1, E2])
        targets_train.append([targets1, targets2])
    return (data_train, targets_train, N_samples)

import os
import math
from sklearn.metrics import cohen_kappa_score
import numpy as np


def batch_generator(sample_list, batch_size):
    for i in range(0, len(sample_list), batch_size):
        yield sample_list[i:i + batch_size]


def create_tmp_dirs(dir_list):
    for d in dir_list:
        os.makedirs(d, exist_ok=True)


def kappa_metric(y_true, y_pred, n_cl=4):
    y = np.array(y_true)
    y_ = np.array(y_pred)
    res = []
    for c in range(n_cl):
        res.append(cohen_kappa_score(y == c, y_ == c))
    return np.array(res)

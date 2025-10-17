from ntpath import isfile

from sympy.printing.pretty.pretty_symbology import line_width
import raybnn_python
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
from io import StringIO

from sklearn.datasets import load_iris as sklearn_load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.datasets import fetch_openml


def main():


    ## Load MNIST dataset

    def load_mnist():
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X=X.astype(np.float32) / 255.0

        x_train = X[:60000].reshape(-1, 28, 28)
        y_train = y[:60000]
        x_test = X[60000:].reshape(-1, 28, 28)
        y_test = y[60000:]



        return x_train, y_train, x_test, y_test

    x_train, y_train, x_test, y_test = load_mnist()

    #Normalize MNIST and Fashion-MNIST dataset, keep IRIS unchanged
    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value)/(max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value)/(max_value - min_value)

    # print("x_test after normalize", x_test)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    ## Parameter setting for Fashion and MNIST dataset
    dir_path = "/tmp/"
    max_input_size = 784
    input_size = 784

    max_output_size = 10
    output_size = 10

    max_neuron_size = 2000

    batch_size = 1000
    traj_size = 1

    proc_num = 2
    active_size = 1000

    training_samples = 60
    crossval_samples = 60
    testing_samples = 10


    #Create Neural Network
    arch_search = raybnn_python.create_start_archtecture(
        input_size,
        max_input_size,

        output_size,
        max_output_size,

        active_size,
        max_neuron_size,

        batch_size,
        traj_size,

        proc_num,
        dir_path
    )

    sphere_rad = arch_search["neural_network"]["netdata"]["sphere_rad"]

    arch_search = raybnn_python.add_neuron_to_existing3(
        10,
		10000,
		sphere_rad/1.3,
		sphere_rad/1.3,
		sphere_rad/1.3,

        arch_search,
    )

    arch_search = raybnn_python.select_forward_sphere(arch_search)

    raybnn_python.print_model_info(arch_search)


    stop_strategy = "STOP_AT_TRAIN_LOSS"
    lr_strategy = "SHUFFLE_CONNECTIONS"
    lr_strategy2 = "MAX_ALPHA"

    loss_function = "sigmoid_cross_entropy_5"

    max_epoch = 10000
    stop_epoch = 100000
    stop_train_loss = 0.005

    max_alpha = 0.01

    exit_counter_threshold = 100000
    shuffle_counter_threshold = 200

    max_input_size = 1176  # CNN features: 24 * 7 * 7
    input_size = 1176

    max_output_size = 10
    output_size = 10
    train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
    train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)
    # print("Train X shape: ", train_x.shape)
    # print("Train Y shape: ", train_y.shape)
    

    class End_to_end_forward(nn.Module):
        def __init__(self):
            super(End_to_end_forward, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
            self.drop = nn.Dropout2d(p=0.2)

        def combine_forward(self, raw_images, verbose=False):
            # First convolutional layer + pooling
            x = F.relu(self.pool(self.conv1(raw_images)))
            if verbose:
                print("After conv1 + pool:", x.shape)
            
            # Second conv layer + pooling
            x = F.relu(self.pool(self.conv2(x)))
            if verbose:
                print("After conv2 + pool:",x.shape)

            # Third conv layer + dropout
            x = F.relu(self.drop(self.conv3(x)))
            if verbose:
                print("After conv3 + drop:", x.shape)

            features = F.dropout(x, training = self.training)
            if verbose:
                print("After dropout: ", features.shape)
            
            features_flat = features.reshape(features.size(0), -1)
            if verbose:
                print("After flattening: ", features_flat.shape)
            features_np = features_flat.detach().cpu().numpy()
            ## Format MNIST dataset
            for i in range(features_np.shape[0]): # 0 -> 60000
                j = (i % batch_size) # 1000
                k = int(i/batch_size) # 60

                train_x[:, j , 0, k ] = features_np[i,:]

                idx = int(y_train[i])
                train_y[idx , j , 0, k ] = 1.0    
            combine_output = raybnn_python.state_space_forward_batch(train_x, train_y, traj_size, max_epoch, arch_search)
            
            
            return combine_output # flatten

    x_train_tensor = torch.from_numpy(x_train).float()
    x_train_tensor = x_train_tensor.unsqueeze(1)
    model_testing = End_to_end_forward()
    output = model_testing.combine_forward(x_train_tensor, verbose=True)


    arch_search = raybnn_python.state_space_forward_batch(
        train_x,
        train_y,
        traj_size,
        max_epoch,
        # proc_num,
        arch_search
        # print out Internal state matrix here
    )

    print("Done without errors!")

if __name__ == '__main__':
    main()
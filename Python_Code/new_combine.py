from pyexpat import features
import numpy as np
import raybnn_python
import mnist
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def main():

    # Load MNIST dataset
    if os.path.isfile("./train-labels-idx1-ubyte.gz") == False:
        mnist.init()

    x_train, y_train, x_test, y_test = mnist.load()

    # Normalize MNIST dataset
    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value)/(max_value-min_value)
    x_test = (x_test.astype(np.float32) - mean_value)/(max_value-min_value)

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

    # Format MNIST dataset
    train_x = np.zeros((input_size,batch_size,traj_size,training_samples))
    train_y = np.zeros((output_size,batch_size,traj_size,training_samples))

    for i in range(x_train.shape[0]):
        j = (i%batch_size)
        k = int(i/batch_size)

        train_x[:, j , 0, k ] = x_train[i,:]

        idx = y_train[i]
        train_y[idx, j, 0, k] = 1.0

    crossval_x = np.copy(train_x)
    crossval_y = np.copy(train_y)

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

    max_epoch = 100000
    stop_epoch = 100000
    stop_train_loss = 0.005

    max_alpha = 0.01

    exit_counter_threshold = 100000
    shuffle_counter_threshold = 200

class CNN_FeatureExtractor(nn.Module):
    # Constructor
    def __init__(self):
        super(CNN_FeatureExtractor).__init__()

    # input images are grayscale, so input channels = 1
        self.conv1 = nn.Conv2d(in_channels=1, 
        out_channels=12, kernel_size=3, stride=1, padding=1)  

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12,
        kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, 
        kernel_size=3, stride=1, padding=1)

        self.drop = nn.Dropout2d(p=0.2)

        # No FC layer bc we just need feature extraction

    # What is x?
    def forward(self, x, verbose=False):
        # First Conv layer + pooling
        x = F.relu(self.pool(self.conv1(x)))
        if verbose:
            print("After conv1 + pool:", x.shape)

        # Second conv layer + pooling
        x = F.relu(self.pool(self.conv2(x)))
        if verbose:
            print("After conv2 + pool:", x.shape)

        # Dropout during training
        features = F.dropout(x, training=self.training)
        if verbose:
            print("After dropout:", features.shape)

        # Return the features directly
        return features

class IntegratedModel:
          
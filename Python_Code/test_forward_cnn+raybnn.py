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
import dataset
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

    ## Load Fashion-MNIST

    # def load_fashion_mnist():
    #     X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

    #     x_train = X[:60000].reshape(-1, 28, 28)
    #     y_train = y[:60000]
    #     x_test = X[60000:].reshape(-1, 28, 28)
    #     y_test = y[60000:]
    #     return x_train, y_train, x_test, y_test
    
    # x_train, y_train, x_test, y_test = load_fashion_mnist()


    # ## Load IRIS dataset
    # def load_iris():
    #     iris = sklearn_load_iris()
    #     X = iris.data
    #     y = iris.target

    #     # Split into train/test (70/30)
    #     x_train, x_test, y_train, y_test = train_test_split(X, y, 
    #     test_size=0.3, random_state=42, stratify=y
    # )
    #     # Reshape (add batch dims and 4th dims)
    #     # reshape to [features, samples, 1, 1]
    #     x_train = x_train.T.reshape(4, -1, 1, 1)
    #     x_test = x_test.T.reshape(4, -1, 1, 1)

    #     x_train = x_train.astype(np.float32)
    #     x_test = x_test.astype(np.float32)
    #     y_train = y_train.astype(np.float32)
    #     y_test = y_test.astype(np.float32)
        
    #     return x_train, y_train, x_test, y_test
    
    # x_train, y_train, x_test, y_test = load_iris()

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

    # print("x_test before normalize", x_test.shape)
    # print("y_test", y_test)

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

    # ## IRIS dataset parameters setting

    # dir_path = "/tmp/"
    # max_input_size = 4
    # input_size = 4

    # max_output_size = 3
    # output_size = 3

    # max_neuron_size = 2000

    # batch_size = 5
    # traj_size = 1

    # proc_num = 2
    # active_size = 1000

    # training_samples = 21
    # crossval_samples = 9
    # testing_samples = 9

    # Initialize train_x, train_y numpy array with 0s


    # # Format IRIS dataset
    # for i in range(x_train.shape[1]):  # x_train.shape[1] = 60000 samples
    #     j = (i % batch_size)
    #     k = int(i/batch_size)

    #     train_x[:, j, 0, k] = x_train[:, i].flatten()  # Get all features for sample i

    #     idx = int(y_train[i])
    #     train_y[idx, j, 0, k] = 1.0

    ## Format MNIST dataset


    # crossval_x = np.copy(train_x)
    # crossval_y = np.copy(train_y)

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

    print("Output keys: ", output.keys())
    print("Output values: ", output.values())
    print("Output items: ", output.items())
    print("Output: ", output)


    # arch_search = raybnn_python.state_space_forward_batch(
    #     train_x,
    #     # input_size,
    #     # max_input_size,

    #     # output_size,
    #     # max_output_size,

    #     # batch_size,
    #     traj_size,
    #     max_epoch,
    #     # proc_num,
    #     arch_search
    #     # print out Internal state matrix here
    # )

    # arch_search = raybnn_python.state_space_forward_batch(
    #     train_x,
    #     train_y,
    #     traj_size,
    #     max_epoch,
    #     # proc_num,
    #     arch_search
    #     # print out Internal state matrix here
    # )
    def plot_loss_from_log(log_file_path):
        """Parse log file and plot loss values with better visualization"""
        loss_values = []
        epochs = []
        
        try:
            with open(log_file_path, 'r') as f:
                for line in f:
                    if 'Epoch' in line and 'Loss' in line:
                        epoch_match = re.search(r'Epoch (\d+)', line)
                        loss_match = re.search(r'Loss = ([\d.]+)', line)
                        
                        if epoch_match and loss_match:
                            epochs.append(int(epoch_match.group(1)))
                            loss_values.append(float(loss_match.group(1)))
            
            if not loss_values:
                print(f"No loss data found in {log_file_path}")
                return
            
            # Create multiple subplots for better visualization
            fig, ax3 = plt.subplots(figsize=(15, 12))
            

            
            # Plot 3: Log scale if loss values are very small
            if min(loss_values) > 0:
                ax3.semilogy(epochs, loss_values, linewidth=1, color='green', alpha=0.8)
                ax3.set_title('Loss CNN+RayBNN')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss (log scale)')
            else:
                # If negative values, show absolute values
                ax3.plot(epochs, np.abs(loss_values), linewidth=1, color='green', alpha=0.8)
                ax3.set_title('Absolute Loss Values')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('|Loss|')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_loss_cnn+raybnn.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print statistics
            print(f"Loss Statistics:")
            print(f"  Initial Loss: {loss_values[0]:.6f}")
            print(f"  Final Loss: {loss_values[-1]:.6f}")
            print(f"  Min Loss: {min(loss_values):.6f}")
            print(f"  Max Loss: {max(loss_values):.6f}")
            print(f"  Mean Loss: {np.mean(loss_values):.6f}")
            print(f"  Std Loss: {np.std(loss_values):.6f}")
            print(f"  Total Epochs: {len(loss_values)}")
            
        except FileNotFoundError:
            print(f"Log file {log_file_path} not found")
        except Exception as e:
            print(f"Error parsing log file: {e}")
    
    log_file = r"/home/hbui/Downloads/RayBNN_Python/Python_Code/print_forward_pass_cnn+raybnn.txt"  
    plot_loss_from_log(log_file)
    print("Done without errors!")

if __name__ == '__main__':
    main()
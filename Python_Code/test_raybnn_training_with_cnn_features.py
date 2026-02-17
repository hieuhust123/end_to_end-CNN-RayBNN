import raybnn_python
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
from io import StringIO

#from sklearn.datasets import load_iris as sklearn_load_iris
#from sklearn.model_selection import train_test_split
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

    # Normalize MNIST dataset
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

    max_epoch = 200000
    stop_epoch = 1000000
    stop_train_loss = 0.005

    max_alpha = 0.005

    exit_counter_threshold = 200000
    shuffle_counter_threshold = 200

    #max_input_size = 1176  # CNN features: 24 * 7 * 7
    #input_size = 1176
    train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
    train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)


    class CaptureStdoutToFile:
        def __init__(self, filename):
            self.filename = filename
            self.stdout_fd = sys.stdout.fileno()
            self.saved_stdout_fd = os.dup(self.stdout_fd)
            self.f = None

        def __enter__(self):
            # Flush any buffered output
            sys.stdout.flush()
            # Open the log file
            self.f = open(self.filename, 'w')
            # Replace stdout with the log file
            os.dup2(self.f.fileno(), self.stdout_fd)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            # Flush and restore stdout
            sys.stdout.flush()
            os.dup2(self.saved_stdout_fd, self.stdout_fd)
            os.close(self.saved_stdout_fd)
            if self.f:
                self.f.close()

    class Training_with_train_network(nn.Module):
        def __init__(self, pretrained_path = None):
            super(Training_with_train_network, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, 
            kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=64, # out=12
            kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=16, # out=24
            kernel_size=3, stride=1, padding=1)
            self.drop = nn.Dropout2d(p=0.1)

            if pretrained_path and os.path.isfile(pretrained_path):
                state_dict = torch.load(pretrained_path)
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained CNN features from {pretrained_path}")
            else:
                print("Using random CNN initialization")

        def cnn(self, raw_images, y_train, y_test, verbose=False):
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

            print("train x shape: ", train_x.shape)
            print("train y shape: ", train_y.shape)

            crossval_x = np.copy(train_x)
            crossval_y = np.copy(train_y)
            
            log_file = "output_raybnn_training_with_cnn_features.txt"
            print(f"Starting training (output redirected to {log_file})")

            with CaptureStdoutToFile(log_file):
                train_network_output = raybnn_python.train_network(
                    train_x,
                    train_y,
                    crossval_x,
                    crossval_y,
                    stop_strategy,
                    lr_strategy,
                    lr_strategy2,
                    loss_function,
                    max_epoch,
                    stop_epoch,
                    stop_train_loss,
                    max_alpha,
                    exit_counter_threshold,
                    shuffle_counter_threshold,
                    arch_search
                )
            test_x = np.zeros((input_size,batch_size,traj_size,testing_samples)).astype(np.float32)
            test_y = np.zeros((output_size,batch_size,traj_size,testing_samples)).astype(np.float32)
            print(f"test_x shape: {test_x.shape}")
            print(f"test_y shape: {test_y.shape}")
            print(f"x_test.shape[0]: {x_test.shape[0]}")
            for i in range(x_test.shape[0]):
                j = (i % batch_size)
                k = int(i/batch_size)

                if k >= testing_samples:
                    print(f"WARNING: k={k} exceeds testing_samples={testing_samples}")
                    break

                test_x[:, j , 0, k ] = x_test[i,:].flatten()
                idx = int(y_test[i])

                if idx < 0 or idx >= output_size:
                    print(f"ERROR: Invalid label {idx} for sample {i}")
                    continue

                test_y[idx , j , 0, k ] = 1.0
            
            print("--- Test data prepared. Starting inference ---\n")

            #Test Neural Network
            output_y = raybnn_python.test_network(
                test_x,

                arch_search
            )
            pred = []
            for i in range(x_test.shape[0]):
                j = (i % batch_size)
                k = int(i/batch_size)

                sample = output_y[:, j , 0, k ]
                #print(sample)

                pred.append(np.argmax(sample))

            y_test = y_test.astype(int)  # Before passing to metrics

            acc = accuracy_score(y_test, np.array(pred).astype(int))

            ret = precision_recall_fscore_support(y_test, pred, average='macro')

            print("Accuracy: ",acc)
            print("Precision: ",ret[0])
            print("Recall: ",ret[1])
            print("F1 Score: ",ret[2])
            print("Support: ",ret[3])

            print("Training finished. Parsing logs...")
            
            # Parse and plot loss
            loss_history = []
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        # Expected format: "Train loss: 0.91234, alpha0: 0.01, i: 100"
                        match = re.search(r"Train loss:\s*([\d\.]+)", line)
                        if match:
                            loss_history.append(float(match.group(1)))

                if loss_history:
                    final_loss = loss_history[-1]
                    print(f"Final training loss: {final_loss:.4f}")
                    plt.figure(figsize=(10, 6))
                    plt.plot(loss_history, label='Training Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Cross Entropy Loss')
                    plt.title('Training Loss of RayBNN model with CNN features input')
                    plt.text(
                        0.02,
                        0.95,
                        f'Final loss: {final_loss:.6f}',
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                    )
                    plt.legend()
                    plt.grid(True)
                    plot_filename = "training_loss_raybnn_with_cnn_features_plot.png"
                    plt.savefig(plot_filename)
                    print(f"Loss plot saved to {plot_filename}")
                else:
                    print("No loss values found in the log file.")
            except Exception as e:
                print(f"Error parsing log file or plotting: {e}")

            return train_network_output # flatten



    x_train_tensor = torch.from_numpy(x_train).float()
    x_train_tensor = x_train_tensor.unsqueeze(1)
    #feature_extractor = Training_with_train_network(pretrained_path = 'cnn_pretrained_conv_only')
    feature_extractor = Training_with_train_network(pretrained_path = None)
    features = feature_extractor.cnn(x_train_tensor, y_train, y_test,verbose=False)

    
    # class Training_with_modified_structure_raybnn(nn.Module):
    #     def __init__(self):
    #         super(Training_with_modified_structure_raybnn, self).__init__()

    #         self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
    #         self.pool = nn.MaxPool2d(kernel_size=2)
    #         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
    #         self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
    #         self.drop = nn.Dropout2d(p=0.2)
            
    #     def cnn(self, raw_images, verbose=False):
    #         # First convolutional layer + pooling
    #         x = F.relu(self.pool(self.conv1(raw_images)))
    #         if verbose:
    #             print("After conv1 + pool:", x.shape)
            
    #         # Second conv layer + pooling
    #         x = F.relu(self.pool(self.conv2(x)))
    #         if verbose:
    #             print("After conv2 + pool:",x.shape)

    #         # Third conv layer + dropout
    #         x = F.relu(self.drop(self.conv3(x)))
    #         if verbose:
    #             print("After conv3 + drop:", x.shape)

    #         features = F.dropout(x, training = self.training)
    #         if verbose:
    #             print("After dropout: ", features.shape)
            
    #         features_flat = features.reshape(features.size(0), -1)
    #         if verbose:
    #             print("After flattening: ", features_flat.shape)
    #         features_np = features_flat.detach().cpu().numpy()
    #         ## Format MNIST dataset
    #         for i in range(features_np.shape[0]): # 0 -> 60000
    #             j = (i % batch_size) # 1000
    #             k = int(i/batch_size) # 60

    #             train_x[:, j , 0, k ] = features_np[i,:]

    #             idx = int(y_train[i])
    #             train_y[idx , j , 0, k ] = 1.0    

    #         print("train x shape: ", train_x.shape)
    #         print("train y shape: ", train_y.shape)

    #         crossval_x = np.copy(train_x)
    #         crossval_y = np.copy(train_y)


    #         train_network_output = raybnn_python.train_network(
    #     train_x,
    #     train_y,

    #     crossval_x,
    #     crossval_y,

    #     stop_strategy,
    #     lr_strategy,
    #     lr_strategy2,

    #     loss_function,
      
    #     max_epoch,
    #     stop_epoch,
    #     stop_train_loss,

    #     max_alpha,
      
    #     exit_counter_threshold,
    #     shuffle_counter_threshold,

    #     arch_search
    # )
            
            
    #         return train_network_output # flatten

    print("Done without errors!")

    # Need to plot a figure here
if __name__ == '__main__':
    main()
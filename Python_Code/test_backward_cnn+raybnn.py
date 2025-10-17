from ntpath import isfile
from tabnanny import verbose

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

# import autograd_end_to_end
# from autograd_end_to_end import AutogradEndToEndModel, test_autograd_gradient_flow


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

    # Convert labels to integers (fix for numpy.object_ error)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

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

    x_train_tensor = torch.from_numpy(x_train).float().unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.from_numpy(y_train).long()
    x_test_tensor = torch.from_numpy(x_test).float().unsqueeze(1)
    y_test_tensor = torch.from_numpy(y_test).long()

    print("x_train_tensor: ", x_train_tensor.shape)
    print("y_train_tensor: ", y_train_tensor.shape)

            
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
    
    
    class CNN(nn.Module):
        def __init__(self):
            super (CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, 
            kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, 
            kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=12, out_channels=24,
            kernel_size=3, stride=1, padding=1)
            self.drop = nn.Dropout2d(p=0.2)  # Added missing dropout layer


        def forward(self, raw_images, y_labels, verbose=True):
            # First convolutional layer + pooling + ReLU
            x = F.relu(self.pool(self.conv1(raw_images)))
            if verbose:
                print("After conv1 + pool + ReLu: ", x.shape)

            # Second conv layer + pooling + ReLU
            x = F.relu(self.pool(self.conv2(x)))
            if verbose:
                print("After conv2 + pool + ReLU: ", x.shape)
            
            # Third conv layer + dropout + ReLU
            x = F.relu(self.drop(self.conv3(x)))
            if verbose:
                print("After conv3 + drop + ReLU: ", x.shape)

            features = F.dropout(x, training=self.training)
            if verbose:
                print("After dropout: ", features.shape)

            features_flat = features.reshape(features.size(0), -1)
            if verbose:
                print("After flattening: ", features_flat.shape)
            
            return features_flat
            ## then call object of this class in AutoGrad class --> connect it to forward pass

    class AutoGradEndtoEnd(torch.autograd.Function):

        @staticmethod
        def forward(ctx, features, y_labels, arch_search, batch_size, 
        traj_size, max_epoch, input_size, output_size, training_samples):
            """
            
            """
            # Save tensors that will be needed in backward
            ctx.arch_search = arch_search
            ctx.batch_size = batch_size
            ctx.traj_size = traj_size
            ctx.max_epoch = max_epoch
            ctx.input_size = input_size
            ctx.output_size = output_size
            ctx.training_samples = training_samples

            print(f"[AUTOGRAD FORWARD] Features shape: {features.shape}")
            print(f"[AUTOGRAD FORWARD] Labels shape: {y_labels.shape}")

            # Convert X and Y to numpy arrays
            features_np = features.detach().cpu().numpy()
            y_labels_np = y_labels.detach().cpu().numpy()

            #input_size = features_np.shape[1]

            # Create training arrays using existing format
            train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
            train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)
            print("train X shape: ", train_x.shape)
            # Divide raw dataset into correspond batches
            for i in range(features_np.shape[0]):
                j = (i% batch_size)
                k = int(i/batch_size)

                train_x[:,j,0,k] = features_np[i,:]
                idx = int(y_labels_np[i])
                if idx < output_size:
                    train_y[idx, j, 0, k] = 1.0

            result = raybnn_python.state_space_forward_batch(train_x, train_y, 
            traj_size, max_epoch, arch_search)

            #--> result: dict???
            
            print("return of RayBNN forward pass: ", type(result))
           
            Yhat_array = np.array(result).astype(np.float32)
            # obj_arch_search = np.array(obj_arch_search).astype(np.float32)

            ctx.save_for_backward(features, y_labels)

            # Convert to Pytorch tensors from numpy
            Yhat_tensor = torch.from_numpy(Yhat_array).to(features.device)

            print("Yhat_tensor shape: ", Yhat_tensor.shape)
            Yhat = Yhat_tensor.squeeze(-1).squeeze(-1).T
            print("Reshaped Yhat: ", Yhat.shape)
            return Yhat

        @staticmethod
        def backward(ctx, grad_output):
            features, y_labels = ctx.saved_tensors
            arch_search = ctx.arch_search
            batch_size = ctx.batch_size
            traj_size = ctx.traj_size
            max_epoch = ctx.max_epoch
            input_size = ctx.input_size
            output_size = ctx.output_size
            training_samples = ctx.training_samples

            features_np = features.detach().cpu().numpy()
            y_labels_np = y_labels.detach().cpu().numpy()

            print("features_np shape: ", features_np.shape)
            print("y_label shape: ", y_labels_np.shape)
            #input_size = features_np.shape[1]

            # Create training arrays using the same format as forward pass
            train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
            train_y = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)

            # Format data the same way as forward pass (FIXED INDENTATION)
            for i in range(features_np.shape[0]):
                j = (i % batch_size)
                k = int(i/batch_size)

                train_x[:, j, 0, k] = features_np[i, :]
                idx = int(y_labels_np[i])
                if idx < output_size:
                    train_y[idx, j, 0, k] = 1.0
            
            # Call RayBNN backward pass
            try:
                print(f"[AUTOGRAD BACKWARD] Calling RayBNN backward with train_x shape: {train_x.shape}")
                grad_result = raybnn_python.state_space_backward_group2(
                    train_x, train_y, traj_size, max_epoch, arch_search
                )
                if grad_result is None:
                    raise RuntimeError("backward returned None (check shapes/devices/NaNs)")

                # PROBLEMS START FROM HERE!!!!!!
                grad_features = grad_output.view(features.shape)
                print(f"[AUTOGRAD BACKWARD] Using pass-through gradients")

            except Exception as e:
                print(f"[AUTOGRAD BACKWARD] Error calling RayBNN backward: {e}")
                # Fallback: return pass-through gradients
                grad_features = grad_output.view(features.shape)
                print("[AUTOGRAD BACKWARD] Using fallback pass-through gradients")
            
            print(f"[AUTOGRAD BACKWARD] Grad features shape: {grad_features.shape}")

            return grad_features, None,None,None,None,None,None

    class EndtoEndTrainer (nn.Module):
        def __init__(self, arch_search, batch_size, traj_size, max_epoch, input_size, output_size, training_samples):
            super().__init__()
            print("1")
            self.cnn = CNN()
            # network_params = obj_arch_search["neural_network"]["network_params"]
            # print(f"network_params type: {type(network_params)}")
            # print(f"network_params keys: {network_params.keys()}")
            # self.raybnn_params = nn.Parameter(torch.from_numpy(network_params))
            
            # Store RayBNN parameters for AutoGradEndtoEnd
            self.arch_search = arch_search
            self.batch_size = batch_size
            self.traj_size = traj_size
            self.max_epoch = max_epoch
            self.input_size = input_size
            self.output_size = output_size
            self.training_samples = training_samples
            print("2")
        def forward(self, raw_images, y_labels, verbose=False):
        # Step 1: CNN forward pass using your existing CNN class
            features = self.cnn(raw_images, y_labels, verbose)
            print("features shape: ", features.shape)
            print("label shape: ",y_labels.shape)
        # Step 2: RayBNN forward pass using your AutoGradEndtoEnd class
            output = AutoGradEndtoEnd.apply(
                features,           # CNN features
                y_labels,          # labels
                self.arch_search,  # RayBNN params
                self.batch_size,   # batch size
                self.traj_size,    # trajectory size
                self.max_epoch,    # max epochs
                self.input_size,    # input size
                self.output_size,   # output size
                self.training_samples 
            )
            print("output shape: ", output.shape)
            return output    



    end_to_end_model = EndtoEndTrainer(arch_search, batch_size, traj_size, max_epoch, input_size, output_size, training_samples)
    
    return end_to_end_model, x_train_tensor, y_train_tensor


def train_ete_model(model, x_train, y_train,batch_size,  num_epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    print("x train shape: ",x_train.shape)
    print("y train shape: ", y_train.shape)
    model.train()

    for epoch in range(num_epoch):
        for i in range(0, min(1000, len(x_train)), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            print("batch x: ",batch_x.shape) # torch.Size([1000, 1, 28, 28])
            print("batch y: ",batch_y.shape) # torch.Size([1000])
            optimizer.zero_grad()

            output = model(batch_x, batch_y, verbose=False) # output shape:  torch.Size([2000, 1000, 2, 1])
            print("debug")
            print("output shape: ", output.shape)
            loss = criterion(output, batch_y)

            loss.backward()

            optimizer.step()

            if i % 500 ==0 :
                print(f'Epoch {epoch+1}/{num_epoch}, Batch {i//batch_size}, Loss: {loss.item():.6f}')

    return model

    # class End_to_end_forward(nn.Module):
    #     def __init__(self):
    #         super(End_to_end_forward, self).__init__()

    #         self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
    #         self.pool = nn.MaxPool2d(kernel_size=2)
    #         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
    #         self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
    #         self.drop = nn.Dropout2d(p=0.2)

    #     def combine_forward(self, raw_images, verbose=False):
    #         # First convolutional layer + pooling + ReLU
    #         x = F.relu(self.pool(self.conv1(raw_images)))

    #         if verbose:
    #             print("After conv1 + pool + ReLU:", x.shape)
            
    #         # Second conv layer + pooling + ReLU
    #         x = F.relu(self.pool(self.conv2(x)))
    #         if verbose:
    #             print("After conv2 + pool + ReLU:",x.shape)

    #         # Third conv layer + dropout + ReLU
    #         x = F.relu(self.drop(self.conv3(x)))
    #         if verbose:
    #             print("After conv3 + drop + ReLU:", x.shape)

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
    #         combine_output = raybnn_python.state_space_forward_batch(train_x, train_y, traj_size, max_epoch, arch_search)
            
            
    #         return combine_output # flatten

    
    # # output = model_testing.combine_forward(x_train_tensor, verbose=True)

    # def train_model():
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #     # Data preparation
    #     x_train_tensor = torch.from_numpy(x_train).float()
    #     x_train_tensor = x_train_tensor.unsqueeze(1)

    #     # Create model
    #     model_testing = End_to_end_forward()
    #     num_training_run = 5

    #     for run in range(num_training_run):
    #         print(f"\n=== Training Run {run + 1}/{num_training_run} ===")

    #         model_testing.train()

    #         with torch.no_grad():
    #             output = model_testing.combine_forward(x_train_tensor, verbose=True)

    #         print(f"Training run {run + 1} completed")
    #         #print(f"Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")

    #     return model_testing

    # trained_model = train_model()
    # # arch_search = raybnn_python.state_space_forward_batch(
    # #     train_x,
    # #     # input_size,
    # #     # max_input_size,

    # #     # output_size,
    # #     # max_output_size,

    # #     # batch_size,
    # #     traj_size,
    # #     max_epoch,
    # #     # proc_num,
    # #     arch_search
    # #     # print out Internal state matrix here
    # # )

    # # arch_search = raybnn_python.state_space_forward_batch(
    # #     train_x,
    # #     train_y,
    # #     traj_size,
    # #     max_epoch,
    # #     # proc_num,
    # #     arch_search
    # #     # print out Internal state matrix here
    # # )
    # # def plot_loss_from_log(log_file_path):
    # #     """Parse log file and plot loss values with better visualization"""
    # #     loss_values = []
    # #     epochs = []
        
    # #     try:
    # #         with open(log_file_path, 'r') as f:
    # #             for line in f:
    # #                 if 'Epoch' in line and 'Loss' in line:
    # #                     epoch_match = re.search(r'Epoch (\d+)', line)
    # #                     loss_match = re.search(r'Loss = ([\d.]+)', line)
                        
    # #                     if epoch_match and loss_match:
    # #                         epochs.append(int(epoch_match.group(1)))
    # #                         loss_values.append(float(loss_match.group(1)))
            
    # #         if not loss_values:
    # #             print(f"No loss data found in {log_file_path}")
    # #             return
            
    # #         # Create multiple subplots for better visualization
    # #         fig, ax3 = plt.subplots(figsize=(15, 12))
            

            
    # #         # Plot 3: Log scale if loss values are very small
    # #         if min(loss_values) > 0:
    # #             ax3.semilogy(epochs, loss_values, linewidth=1, color='green', alpha=0.8)
    # #             ax3.set_title('Loss CNN+RayBNN')
    # #             ax3.set_xlabel('Epoch')
    # #             ax3.set_ylabel('Loss (log scale)')
    # #         else:
    # #             # If negative values, show absolute values
    # #             ax3.plot(epochs, np.abs(loss_values), linewidth=1, color='green', alpha=0.8)
    # #             ax3.set_title('Absolute Loss Values')
    # #             ax3.set_xlabel('Epoch')
    # #             ax3.set_ylabel('|Loss|')
    # #         ax3.grid(True, alpha=0.3)
            
    # #         plt.tight_layout()
    # #         plt.savefig('training_loss_cnn+raybnn.png', dpi=300, bbox_inches='tight')
    # #         plt.show()
            
    # #         # Print statistics
    # #         print(f"Loss Statistics:")
    # #         print(f"  Initial Loss: {loss_values[0]:.6f}")
    # #         print(f"  Final Loss: {loss_values[-1]:.6f}")
    # #         print(f"  Min Loss: {min(loss_values):.6f}")
    # #         print(f"  Max Loss: {max(loss_values):.6f}")
    # #         print(f"  Mean Loss: {np.mean(loss_values):.6f}")
    # #         print(f"  Std Loss: {np.std(loss_values):.6f}")
    # #         print(f"  Total Epochs: {len(loss_values)}")
            
    # #     except FileNotFoundError:
    # #         print(f"Log file {log_file_path} not found")
    # #     except Exception as e:
    # #         print(f"Error parsing log file: {e}")
    
    # # log_file = r"/home/hbui/Downloads/RayBNN_Python/Python_Code/print_forward_pass_cnn+raybnn.txt"  
    # # plot_loss_from_log(log_file)
    # # print("Done without errors!")

if __name__ == '__main__':
    end_to_end_model, x_train_tensor, y_train_tensor = main()
    batch_size = 1000
    num_epoch = 5
    # Now call the training function (use the correct name from your code)
    trained_model = train_ete_model(end_to_end_model, x_train_tensor, y_train_tensor,batch_size,  num_epoch)
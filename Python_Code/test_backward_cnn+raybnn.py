


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
import time
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

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)


    #Normalize MNIST and Fashion-MNIST dataset, keep IRIS unchanged
    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value)/(max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value)/(max_value - min_value)

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
    max_input_size = 200
    input_size = 200

    max_output_size = 10
    output_size = 10

    max_neuron_size = 5000

    batch_size = 1000
    traj_size = 1

    proc_num = 2
    active_size = 2500

    training_samples = 60
    crossval_samples = 60
    testing_samples = 10

    alpha0 = 0.000 # 0.001

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

    max_epoch = 5    
    
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
            self.drop = nn.Dropout2d(p=0.1)  # changed to 0.1

            # just added
            self.projection = nn.Linear(1176, 200)
            self.bn = nn.BatchNorm1d(200)

        def forward(self, raw_images, y_labels, verbose=False):
            # First convolutional layer + pooling + ReLU
            x = F.relu(self.pool(self.conv1(raw_images)))
            if verbose:
                print("After conv1 + pool + ReLu: ", x.shape) # torch.Size([1000, 12, 14, 14])

            # Second conv layer + pooling + ReLU
            x = F.relu(self.pool(self.conv2(x)))
            if verbose:
                print("After conv2 + pool + ReLU: ", x.shape) # torch.Size([1000, 12, 7, 7])
            
            # Third conv layer + dropout + ReLU
            x = F.relu(self.drop(self.conv3(x)))
            if verbose:
                print("After conv3 + drop + ReLU: ", x.shape) # torch.Size([1000, 24, 7, 7])

            # Dropout behavior depends on model.training:
            # - train(): randomly zeros ~10% of activations (regularization)
            # - eval(): no dropout, all activations pass through (consistent inference)
            features = F.dropout(x, training=self.training)
            if verbose:
                print("After dropout: ", features.shape) # torch.Size([1000, 24, 7, 7])

            features_flat = features.reshape(features.size(0), -1)
            if verbose:
                print("After flattening: ", features_flat.shape) # torch.Size([1000, 1176])
            
            # Apply projection
            features_reduced = self.projection(features_flat)
            if verbose:
                print("After projection: ", features_reduced.shape) # torch.Size([1000, 200])
            features_normalized = self.bn(features_reduced) # Apply BN here
            return features_normalized

    class AutoGradEndtoEnd(torch.autograd.Function):
        # Class variable to store updated arch_search from backward pass
        _updated_arch_search = None
        _current_epoch = 0
        @staticmethod
        def forward(ctx, features_flat, y_labels, arch_search, batch_size, 
        traj_size, max_epoch, input_size, output_size, training_samples, alpha0):
            """
            
            """
            # Use updated arch_search from previous backward pass if available
            if AutoGradEndtoEnd._updated_arch_search is not None:
                arch_search = AutoGradEndtoEnd._updated_arch_search
                # Note: The model's arch_search will be updated via the class variable
                # The next forward pass will use the updated arch_search
            
            # Save tensors that will be needed in backward
            ctx.arch_search = arch_search
            ctx.batch_size = batch_size
            ctx.traj_size = traj_size
            ctx.max_epoch = max_epoch
            ctx.input_size = input_size
            ctx.output_size = output_size
            ctx.training_samples = training_samples
            ctx.alpha0 = alpha0

            # print(f"[AUTOGRAD FORWARD] Features shape: {features_flat.shape}") # torch.Size([1000, 1176])
            # print(f"[AUTOGRAD FORWARD] Labels shape: {y_labels.shape}") # torch.Size([1000])

            # Convert X and Y to numpy arrays
            features_np = features_flat.detach().cpu().numpy()
            y_labels_np = y_labels.detach().cpu().numpy()

            # Create training arrays using existing format
            train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
            train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)

            print("train X shape: ", train_x.shape) # (1176, 1000, 1, 60)
            # Divide raw dataset into correspond batches
            for i in range(features_np.shape[0]):
                j = (i% batch_size)
                k = int(i/batch_size)

                train_x[:,j,0,k] = features_np[i,:]
                idx = int(y_labels_np[i])
                if idx < output_size:
                    train_y[idx, j, 0, k] = 1.0

            # print("train X shape after feeding input from CNN: ", train_x.shape) # (1176, 1000, 1, 60)
            # print("train Y shape after feeding input from CNN: ", train_y.shape) # (10, 1000, 1, 60)
            result = raybnn_python.state_space_forward_batch(train_x, train_y, 
            traj_size, max_epoch, arch_search)

            # Return of forward pass
            print("return of RayBNN forward pass: ", type(result)) # Numpy array
            
            Yhat_array = np.array(result).astype(np.float32)

            ctx.save_for_backward(features_flat, y_labels)

            # Convert to Pytorch tensors from numpy
            Yhat_tensor = torch.from_numpy(Yhat_array).to(features_flat.device)

            # print("Yhat_tensor shape: ", Yhat_tensor.shape) # torch.Size([10, 1000, 1, 1])
            Yhat = Yhat_tensor.squeeze(-1).squeeze(-1).T
            print("Reshaped Yhat: ", Yhat.shape) # torch.Size([1000, 10])
            
            # TEST 4: RayBNN Output Quality Analysis
            print("\n" + "="*60)
            print("TEST 4: RayBNN Output ")
            print("\n")
            print(f"Output shape: {Yhat.shape}")
            print(f"Output range: [{Yhat.min().item():.6f}, {Yhat.max().item():.6f}]")
            print(f"Output mean: {Yhat.mean().item():.6f}")
            print(f"Output std: {Yhat.std().item():.6f}")
            
            # Check predictions
            preds = Yhat.argmax(dim=1)
            unique_classes = len(torch.unique(preds))
            print(f"Unique predicted classes: {unique_classes} / {Yhat.shape[1]}")
            
            if unique_classes == 1:
                print("⚠️ CRITICAL: RayBNN always predicts same class! (Mode collapse)")
                print("  → RayBNN is stuck - CNN cannot learn from this!")
            elif unique_classes < Yhat.shape[1] * 0.3:
                print(f"⚠️ WARNING: Only {unique_classes} classes predicted (low diversity)")
            else:
                print(f"✓ Good diversity: {unique_classes} different classes predicted")
            
            # Check class distribution
            class_counts = torch.bincount(preds, minlength=Yhat.shape[1])
            print(f"Class distribution: {class_counts.tolist()}")
            
            # Check entropy (diversity measure)
            probs = torch.softmax(Yhat, dim=1)
            print(f"probs: {probs}")
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            max_entropy = np.log(Yhat.shape[1])  # log(10) for 10 classes
            print(f"Output entropy: {entropy.item():.6f} (max possible: {max_entropy:.6f})")
            
            if entropy < 0.5:
                print("⚠️ WARNING: Outputs are too confident/similar (low entropy)")
                print("  → RayBNN outputs lack diversity - may indicate mode collapse")
            elif entropy < max_entropy * 0.5:
                print("⚠️ WARNING: Entropy is below 50% of maximum")
                print("  → RayBNN outputs are somewhat overconfident")
            else:
                print("✓ Good entropy: Outputs have reasonable diversity")
            
            # Check if outputs are reasonable (not all zeros or NaNs)
            if torch.isnan(Yhat).any():
                print("✗ CRITICAL: Output contains NaN values!")
            elif torch.isinf(Yhat).any():
                print("✗ CRITICAL: Output contains Inf values!")
            elif torch.allclose(Yhat, torch.zeros_like(Yhat)):
                print("✗ CRITICAL: Output is all zeros!")
            else:
                print("✓ Output values are valid (no NaN/Inf/zeros)")
            
            # Sample outputs for inspection
            print(f"\nSample outputs (first 5 samples):")
            for i in range(min(5, Yhat.shape[0])):
                pred_class = preds[i].item()
                confidence = probs[i, pred_class].item()
                true_label = y_labels[i].item() if i < y_labels.shape[0] else 'N/A'
                print(f"  Sample {i}: predicted={pred_class}, "
                      f"confidence={confidence:.3f}, "
                      f"true_label={true_label}")
            
            print("="*60 + "\n")
            
            # Store entropy for epoch tracking
            probs = torch.softmax(Yhat, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            # Store in a way that can be accessed by the training loop
            ctx.entropy = entropy.item()
            
            return Yhat

        @staticmethod
        def backward(ctx, grad_output):
            features_flat, y_labels = ctx.saved_tensors
            arch_search = ctx.arch_search
            batch_size = ctx.batch_size
            traj_size = ctx.traj_size
            max_epoch = ctx.max_epoch
            input_size = ctx.input_size
            output_size = ctx.output_size
            training_samples = ctx.training_samples
            alpha0 = ctx.alpha0

            current_epoch = AutoGradEndtoEnd._current_epoch

            features_np = features_flat.detach().cpu().numpy()
            y_labels_np = y_labels.detach().cpu().numpy()

            print("features_np shape: ", features_np.shape)
            print("y_label shape: ", y_labels_np.shape)
            #input_size = features_np.shape[1]

            # Create training arrays using the same format as forward pass
            train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
            train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)

            # Format data the same way as forward pass 
            for i in range(features_np.shape[0]):
                j = (i % batch_size)
                k = int(i/batch_size)

                train_x[:, j, 0, k] = features_np[i, :]
                idx = int(y_labels_np[i])
                if idx < output_size:
                    train_y[idx, j, 0, k] = 1.0
            
            print("CHECKPOINT 1: RayBNN Gradient Output")
            # Call RayBNN backward pass
            try:
                print(f"[AUTOGRAD BACKWARD] Calling RayBNN backward with train_x shape: {train_x.shape}") # (1176, 1000, 1, 60)

                # Call backward pass - now returns (updated_arch_search, grad_result)
                updated_arch_search, grad_result = raybnn_python.state_space_backward_group2(
                    train_x, train_y, traj_size, max_epoch, alpha0, arch_search, current_epoch
                )
                
                # Update arch_search with new parameters
                ctx.arch_search = updated_arch_search
                # Store updated arch_search in class variable for next forward pass
                AutoGradEndtoEnd._updated_arch_search = updated_arch_search
                
                print("[AUTOGRAD BACKWARD] ✓ RayBNN parameters updated!")
                
                # Convert grad_result to numpy array if needed
                if not isinstance(grad_result, np.ndarray):
                    grad_result = np.array(grad_result, dtype=np.float32)
                
                # grad_result = grad_result * 1000.0  # Multiply by 1000x

                has_nan = np.isnan(grad_result).any()
                has_inf = np.isinf(grad_result).any()
                all_zeros = np.allclose(grad_result, 0)

                if has_nan:
                    print(f"⚠️ WARNING: grad_result has NaN values!")
                if has_inf:
                    print(f"⚠️ WARNING: grad_result has Inf values!")
                if all_zeros:
                    print(f"⚠️ WARNING: grad_result is all zeros!")

                print("\n Raw RayBNN Grads")

                print(f"grad_result shape: {grad_result.shape}")
                print(f"grad_result stats: mean={grad_result.mean():.2e}, std={grad_result.std():.2e}")
                print(f"grad_result range: [{grad_result.min():.2e}, {grad_result.max():.2e}]")
                print(f"grad_result norm: {np.linalg.norm(grad_result):.2e}")
                print(f"Non-zero elements: {np.count_nonzero(grad_result)}/{grad_result.size} ({np.count_nonzero(grad_result)/grad_result.size*100:.1f}%)")

                print(f"[AUTOGRAD BACKWARD] features shape: {grad_result.shape}") # (1176, 1000, 1, 1)
                grad_result_reshaped = grad_result[:, :, 0, 0].T
                print(f"[AUTOGRAD BACKWARD] grad_result_reshaped: {grad_result_reshaped.shape}") # (1000, 1176) 
                print(f"[AUTOGRAD BACKWARD] grad_result_reshaped stats: mean={grad_result_reshaped.mean():.2e}, std={grad_result_reshaped.std():.2e}")
                print(f"[AUTOGRAD BACKWARD] grad_result_reshaped norm: {np.linalg.norm(grad_result_reshaped):.2e}")

                grad_features = torch.from_numpy(grad_result_reshaped).to(features_flat.device)
                grad_features = grad_features   # Changed from 0.01 to 0.001   #############
                
                # print("\n STAGE 4: After Safety Clamps")
                # grad_norm_before_clamp = torch.norm(grad_features).item()
                # grad_features = torch.nan_to_num(grad_features, nan=0.0, posinf=0.001, neginf=-0.001)
                # grad_norm_after_nan = torch.norm(grad_features).item()
                # grad_features = torch.clamp(grad_features, -0.01, 0.01)
                # grad_norm_after_clamp = torch.norm(grad_features).item()

                # print(f"Before clamps: norm={grad_norm_before_clamp:.2e}")
                # print(f"After nan_to_num: norm={grad_norm_after_nan:.2e}")
                # print(f"After clamp: norm={grad_norm_after_clamp:.2e}")
                # print(f"Clamp impact: {grad_norm_after_clamp/grad_norm_before_clamp:.2e}x")

                # # Check how many values were clamped
                # clamped_count = ((torch.from_numpy(grad_result_reshaped).to(features_flat.device) * 0.001).abs() > 0.01).sum().item()
                # print(f"Values clamped: {clamped_count}/{grad_features.numel()} ({clamped_count/grad_features.numel()*100:.1f}%)")
                assert grad_features.shape == features_flat.shape, \
                f"Gradient shape {grad_features.shape} doesn't match features {features_flat.shape}"

                # print("CHECKPOINT 2: Pytorch Gradient Tensor")
                # print(f"grad_features shape: {grad_features.shape}")
                # print(f"grad_features device: {grad_features.device}")
                # print(f"grad_features requires_grad: {grad_features.requires_grad}")
                # print(f"grad_features mean: {grad_features.mean().item():.3e}")
                # print(f"grad_features std: {grad_features.std().item():.3e}")
                # print(f"grad_features min: {grad_features.min().item():.3e}")
                # print(f"grad_features max: {grad_features.max().item():.3e}")

                print("\n✓ Backward pass completed!")
            except Exception as e:
                print(f"[AUTOGRAD BACKWARD] Error calling RayBNN backward: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: return pass-through gradients
                grad_features = torch.zeros_like(features_flat)  # Create fallback gradient
                print("[AUTOGRAD BACKWARD] Using zero gradients as fallback")
            
            return grad_features, None,None,None,None,None,None, None, None, None

    class EndtoEndTrainer (nn.Module):
        def __init__(self, arch_search, batch_size, traj_size, max_epoch, input_size, output_size, training_samples, alpha0):
            super().__init__()
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
            self.alpha0 = alpha0

        def update_epoch(self, epoch):
        #Update the current epoch for the autograd function
            AutoGradEndtoEnd._current_epoch = epoch


        def forward(self, raw_images, y_labels, verbose=True):
        # Step 1: CNN forward pass using your existing CNN class
            features = self.cnn(raw_images, y_labels, verbose)
            
            # # 🚀 CRITICAL FIX 3: AGGRESSIVE Feature Standardization + Clamping
            features_std = (features - features.mean()) / (features.std() + 1e-8)  # Standardize
            features_normalized = torch.clamp(features_std, -3.0, 3.0)  # Clamp to [-3, 3]
            # print(f"Applied aggressive feature normalization: {features.shape} -> {features_normalized.shape}")
            # print(f"Original features - mean: {features.mean().item():.6f}, std: {features.std().item():.6f}")
            # print(f"Standardized+Clamped - mean: {features_normalized.mean().item():.6f}, std: {features_normalized.std().item():.6f}")
            
            # print("features shape: ", features_normalized.shape)
            # print("label shape: ",y_labels.shape)
        # Step 2: RayBNN forward pass using your AutoGradEndtoEnd class
            # Update model's arch_search if there's an updated version from backward pass
            if AutoGradEndtoEnd._updated_arch_search is not None:
                self.arch_search = AutoGradEndtoEnd._updated_arch_search
                print("[MODEL] ✓ Updated arch_search with new RayBNN parameters")
            
            # Use the most recent arch_search (may have been updated in previous backward pass)
            output = AutoGradEndtoEnd.apply(
                features_normalized, # NORMALIZED CNN features for stability
                y_labels,          # labels
                self.arch_search,  # RayBNN params (will be updated in backward)
                self.batch_size,   # batch size
                self.traj_size,    # trajectory size
                self.max_epoch,    # max epochs
                self.input_size,    # input size
                self.output_size,   # output size
                self.training_samples,
                self.alpha0
            )
            print("output shape: ", output.shape)
            return output    
        
        def update_arch_search(self, updated_arch_search):
            """Update the model's arch_search with new parameters from backward pass"""
            self.arch_search = updated_arch_search    



    end_to_end_model = EndtoEndTrainer(arch_search, batch_size, traj_size, max_epoch, input_size, output_size, training_samples, alpha0)
    
    return end_to_end_model, x_train_tensor, y_train_tensor


def train_ete_model(model, x_train, y_train,batch_size,  max_epoch):
    # CRITICAL FIX 4: MUCH smaller learning rate for stability
    for param in model.cnn.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training tracking
    epoch_losses = []
    epoch_accuracies = []
    epoch_raybnn_entropies = []
    best_loss = float('inf')
    epoch_times = []  
    patience_counter = 0
    early_stop_patience = 5
    print("x train shape: ",x_train.shape)
    print("y train shape: ", y_train.shape)

    # Store initial CNN parameters
    cnn_params_init = {
        name: param.clone().detach() 
        for name, param in model.cnn.named_parameters()
    }
    
    print("=== Initial CNN Parameters ===")
    for name, param in cnn_params_init.items():
        print(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    # TEST 6: Store initial RayBNN parameters
    print("\n" + "="*60)
    print("TEST 6: RayBNN Parameter Check after extracted from Rust side")
    print("="*60)
    
    try:
        # Extract RayBNN network_params
        network_params_obj = model.arch_search["neural_network"]["network_params"]
        print(f"RayBNN network_params type: {type(network_params_obj)}")
        print(f"RayBNN network_params keys: {network_params_obj.keys() if isinstance(network_params_obj, dict) else 'N/A'}")
        
        # Try to convert to numpy array
        if isinstance(network_params_obj, np.ndarray):
            raybnn_params_before = network_params_obj.copy()
        elif hasattr(network_params_obj, 'to_numpy'):
            # If it's an ArrayFire array with to_numpy method
            raybnn_params_before = network_params_obj.to_numpy().copy()
        elif isinstance(network_params_obj, dict):
            # If it's a dict, try to extract the actual array
            print(f"  Dict keys: {list(network_params_obj.keys())}")
            # Try common keys
            for key in ['data', 'values', 'array', 'params']:
                if key in network_params_obj:
                    temp = network_params_obj[key]
                    print(f"    Found key '{key}', type: {type(temp)}")
                    if isinstance(temp, np.ndarray):
                        raybnn_params_before = temp.copy()
                        print(f"    ✓ Extracted numpy array from '{key}'")
                        break
                    elif isinstance(temp, (list, tuple)):
                        # Convert list/tuple to numpy array
                        try:
                            raybnn_params_before = np.array(temp, dtype=np.float32).copy()
                            # Reshape if shape is provided
                            if 'shape' in network_params_obj:
                                shape = network_params_obj['shape']
                                if isinstance(shape, (list, tuple)):
                                    raybnn_params_before = raybnn_params_before.reshape(shape)
                            print(f"    ✓ Converted list/tuple from '{key}' to numpy array")
                            break
                        except Exception as e:
                            print(f"    ✗ Failed to convert '{key}' to numpy array: {e}")
                    else:
                        print(f"    ✗ '{key}' is not a numpy array or list")
            else:
                # If no standard key worked, try to convert the whole dict
                print("  Attempting to extract array from dict structure...")
                raybnn_params_before = None
        else:
            # Try to convert using numpy
            try:
                raybnn_params_before = np.array(network_params_obj, dtype=np.float32).copy()
            except:
                print(f"  Cannot convert to numpy array")
                raybnn_params_before = None
        
        if raybnn_params_before is not None and isinstance(raybnn_params_before, np.ndarray):
            print(f"✓ Successfully extracted RayBNN network_params")
            print(f"RayBNN network_params shape: {raybnn_params_before.shape}")
            # print(f"RayBNN network_params dtype: {raybnn_params_before.dtype}")
            print(f"RayBNN network_params mean: {raybnn_params_before.mean():.6e}")
            print(f"RayBNN network_params std: {raybnn_params_before.std():.6e}")
            print(f"RayBNN network_params min: {raybnn_params_before.min():.6e}")
            print(f"RayBNN network_params max: {raybnn_params_before.max():.6e}")
        else:
            print("Could not extract network_params as numpy array")
            print(f"  Type: {type(network_params_obj)}")
            print("  Will store the object directly for comparison")
            raybnn_params_before = network_params_obj  # Store as-is for comparison
        
        print("\nWill check if RayBNN parameters change after training...")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Could not extract RayBNN parameters: {e}")
        raybnn_params_before = None
    
    # Set model to training mode
    # This enables dropout (line 186: F.dropout uses self.training)
    # and makes BatchNorm use batch statistics instead of running averages
    model.train()
    print(f"\n{'='*50}")
    print(f"ENHANCED TRAINING: {max_epoch} epochs, batch_size={batch_size}")
    print(f"{'='*50}")

    batch_idx = len(x_train) // batch_size

    for epoch in range(max_epoch):

        model.update_epoch(epoch)
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_entropies = []
        
        print(f"\nEPOCH {epoch+1}/{max_epoch}")
        print("-" * 60)
        
        for i in range(batch_idx):
            start_idx = i * batch_size
            batch_x = x_train[start_idx:start_idx+batch_size]
            batch_y = y_train[start_idx:start_idx+batch_size]

            optimizer.zero_grad()

            # Reduce verbosity - only verbose for first batch of each epoch
            verbose = (i == 0 and epoch < 3)  # Only first 3 epochs, first batch
            output = model(batch_x, batch_y, verbose=verbose)
            loss = criterion(output, batch_y)
            
            # CRITICAL FIX 7: Loss explosion detection
            if loss.item() > 10.0 or torch.isnan(loss) or torch.isinf(loss):
                print(f"LOSS EXPLOSION DETECTED: {loss.item():.3f}")
                print("Stopping training to prevent further damage")
                return model  # Early termination

            loss.backward()  ##################
            # Stage 5: CNN Parameter Gradients
            print("\n CNN Parameter Gradients")
            total_grad_norm = 0
            param_grad_stats = {}

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    total_grad_norm += grad_norm ** 2
                    
                    param_grad_stats[name] = {
                        'norm': grad_norm,
                        'mean': grad_mean,
                        'std': grad_std,
                        'min': param.grad.min().item(),
                        'max': param.grad.max().item()
                    }
                    
                    # Flag problematic gradients
                    status = "✓"
                    if grad_norm < 1e-8:
                        status = "💀 VANISHED"
                    elif grad_norm < 1e-6:
                        status = "⚠️ TINY"
                    elif grad_norm > 1.0:
                        status = "🔥 LARGE"
                        
                    print(f"  {name}: norm={grad_norm:.2e} {status}")

            total_grad_norm = total_grad_norm ** 0.5
            print(f"Total gradient norm: {total_grad_norm:.2e}")

            # # CRITICAL FIX 1: AGGRESSIVE Gradient Clipping 
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # 10x more aggressive

            # # Stage 6: After Gradient Clipping
            # print("\nSTAGE 6: After Gradient Clipping")
            # total_grad_norm_after = 0
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm().item()
            #         total_grad_norm_after += grad_norm ** 2

            # total_grad_norm_after = total_grad_norm_after ** 0.5
            # print(f"Total gradient norm after clipping: {total_grad_norm_after:.2e}")
            # print(f"Clipping impact: {total_grad_norm_after/total_grad_norm:.2e}x")

            # if total_grad_norm_after < 1e-8:
            #     print("💀 CRITICAL: Gradients vanished after clipping!")
            # elif total_grad_norm_after < 1e-6:
            #     print("⚠️ WARNING: Gradients very small after clipping")

            optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            epoch_total += batch_y.size(0)
            epoch_correct += (predicted == batch_y).sum().item()
            
            # Store entropy from the autograd context (calculated in forward pass)
            # We'll extract this from the Test 4 output for now
            # In practice, we could make AutoGradEndtoEnd store this more elegantly
            with torch.no_grad():
                probs = torch.softmax(output, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
                batch_entropies.append(entropy)
            # Progress indicator every 20 batches
            
            current_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            print(f"  Batch {i}: loss={loss.item():.4f}, acc={current_acc:.3f}")

        # End of epoch statistics
        avg_loss = epoch_loss / batch_idx
        epoch_accuracy = epoch_correct / epoch_total
        avg_entropy = sum(batch_entropies) / len(batch_entropies) if batch_entropies else 2.303
        epoch_time = time.time() - epoch_start_time
        
        # Store metrics
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(epoch_accuracy)
        epoch_raybnn_entropies.append(avg_entropy)
        epoch_times.append(epoch_time)
        # Learning rate scheduling
        #scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Enhanced Epoch Summary with Gradient Analysis
        print(f"\nGRADIENT FLOW SUMMARY (Epoch {epoch+1}):")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Loss: {avg_loss:.6f} (change: {avg_loss - epoch_losses[-2] if len(epoch_losses) > 1 else 'N/A'})")
        print(f"  Accuracy: {epoch_accuracy:.4f} ({epoch_accuracy*100:.1f}%)")


        # Learning progress check
        if len(epoch_losses) > 1:
            loss_change = epoch_losses[-2] - avg_loss
            if abs(loss_change) < 1e-6:
                print(f"  LEARNING STATUS: STUCK (loss not changing)")
            elif loss_change > 0:
                print(f"  LEARNING STATUS: IMPROVING (loss decreased by {loss_change:.6f})")
            else:
                print(f"  LEARNING STATUS: DEGRADING (loss increased by {-loss_change:.6f})")

        print(f"  RayBNN Entropy: {avg_entropy:.4f}/2.303 ({(avg_entropy/2.303)*100:.1f}% of max)")
        print(f"  Learning Rate: {current_lr:.6f}")

        # RayBNN learning progress
        if avg_entropy < 2.0:
            print(f"  RayBNN showing learning progress! (entropy < 2.0)")
        elif avg_entropy < 2.2:
            print(f"  RayBNN showing some patterns (entropy < 2.2)")
        else:
            print(f"  RayBNN still exploring (high entropy)")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered (no improvement for {early_stop_patience} epochs)")
            break
            
        # CNN parameter analysis for key epochs      ###################
        if epoch in [0, 1, 4, 9, 14] or epoch == max_epoch - 1:
            print(f"\nPARAMETER UPDATE ANALYSIS (Epoch {epoch+1}):")
            total_param_change = 0
            for name, param in model.named_parameters():
                if name in cnn_params_init:
                    param_change = (param.detach() - cnn_params_init[name]).abs().mean().item()
                    total_param_change += param_change
                    
                    status = "✓"
                    if param_change < 1e-8:
                        status = "💀 NO CHANGE"
                    elif param_change < 1e-6:
                        status = "⚠️ TINY CHANGE"
                        
                    print(f"  {name}: change={param_change:.2e} {status}")

            print(f"  📊 Total parameter change: {total_param_change:.2e}")
        
        print("-" * 60)
    
    # COMPREHENSIVE TRAINING SUMMARY
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE - COMPREHENSIVE SUMMARY")
    print(f"{'='*50}")
    
    print(f"FINAL METRICS:")
    print(f"Epochs completed: {len(epoch_losses)}")
    print(f"Total training time: {sum(epoch_times):.1f}s")
    print(f"Final loss: {epoch_losses[-1]:.4f}")
    print(f"Loss improvement: {epoch_losses[0] - epoch_losses[-1]:.4f} ({((epoch_losses[0] - epoch_losses[-1])/epoch_losses[0]*100):.1f}%)")
    print(f"Final accuracy: {epoch_accuracies[-1]:.4f} ({epoch_accuracies[-1]*100:.1f}%)")
    print(f"Accuracy improvement: {epoch_accuracies[-1] - epoch_accuracies[0]:.4f} ({(epoch_accuracies[-1] - epoch_accuracies[0])*100:.1f} percentage points)")
    print(f"Final RayBNN entropy: {epoch_raybnn_entropies[-1]:.4f}/2.303")
    
    # Learning progress analysis
    print(f"\nLEARNING ANALYSIS:")
    if epoch_losses[0] - epoch_losses[-1] > 0.1:
        print(f"  EXCELLENT: Significant loss reduction ({((epoch_losses[0] - epoch_losses[-1])/epoch_losses[0]*100):.1f}%)")
    elif epoch_losses[0] - epoch_losses[-1] > 0.05:
        print(f"  GOOD: Moderate loss reduction ({((epoch_losses[0] - epoch_losses[-1])/epoch_losses[0]*100):.1f}%)")
    else:
        print(f"  POOR: Minimal loss reduction ({((epoch_losses[0] - epoch_losses[-1])/epoch_losses[0]*100):.1f}%)")
    
    if epoch_accuracies[-1] > 0.5:
        print(f"  EXCELLENT: High accuracy achieved ({epoch_accuracies[-1]*100:.1f}%)")
    elif epoch_accuracies[-1] > 0.3:
        print(f"  GOOD: Decent accuracy ({epoch_accuracies[-1]*100:.1f}%)")
    else:
        print(f"  POOR: Low accuracy ({epoch_accuracies[-1]*100:.1f}%)")
    
    if epoch_raybnn_entropies[-1] < 2.0:
        print(f"  EXCELLENT: RayBNN learned meaningful patterns (entropy {epoch_raybnn_entropies[-1]:.3f})")
    elif epoch_raybnn_entropies[-1] < 2.2:
        print(f"  GOOD: RayBNN showing some learning (entropy {epoch_raybnn_entropies[-1]:.3f})")
    else:
        print(f"  POOR: RayBNN still random-like (entropy {epoch_raybnn_entropies[-1]:.3f})")
    

    
    # TEST 6: Check if RayBNN parameters changed after training
    print("\n" + "="*60)
    print("TEST 6: RayBNN Parameter Update Check (After Training)")
    print("="*60)
    
    if raybnn_params_before is not None:
        try:
            network_params_obj_after = model.arch_search["neural_network"]["network_params"]
            
            # Try to convert after-training params to numpy
            if isinstance(network_params_obj_after, np.ndarray):
                raybnn_params_after = network_params_obj_after.copy()
            elif hasattr(network_params_obj_after, 'to_numpy'):
                raybnn_params_after = network_params_obj_after.to_numpy().copy()
            elif isinstance(network_params_obj_after, dict):
                for key in ['data', 'values', 'array', 'params']:
                    if key in network_params_obj_after:
                        temp = network_params_obj_after[key]
                        if isinstance(temp, np.ndarray):
                            raybnn_params_after = temp.copy()
                            break
                        elif isinstance(temp, (list, tuple)):
                            # Convert list/tuple to numpy array
                            try:
                                raybnn_params_after = np.array(temp, dtype=np.float32).copy()
                                # Reshape if shape is provided
                                if 'shape' in network_params_obj_after:
                                    shape = network_params_obj_after['shape']
                                    if isinstance(shape, (list, tuple)):
                                        raybnn_params_after = raybnn_params_after.reshape(shape)
                                break
                            except:
                                pass
                else:
                    raybnn_params_after = None
            else:
                try:
                    raybnn_params_after = np.array(network_params_obj_after, dtype=np.float32).copy()
                except:
                    raybnn_params_after = None
            
            if isinstance(raybnn_params_after, np.ndarray) and isinstance(raybnn_params_before, np.ndarray):
                # Compare parameters
                param_diff = np.abs(raybnn_params_after - raybnn_params_before)
                mean_change = param_diff.mean()
                max_change = param_diff.max()
                changed_fraction = (param_diff > 1e-6).sum() / param_diff.size
                
                print(f"RayBNN network_params comparison:")
                print(f"  Mean absolute change: {mean_change:.6e}")
                print(f"  Max absolute change: {max_change:.6e}")
                print(f"  Fraction of params changed (>1e-6): {changed_fraction:.3f} ({changed_fraction*100:.1f}%)")
                
                # Check if parameters are identical
                are_identical = np.allclose(raybnn_params_before, raybnn_params_after, atol=1e-6)
                
                print("\n" + "-"*60)
                if are_identical:
                    print("CRITICAL: RayBNN parameters DID NOT CHANGE!")
                    print("  → RayBNN is NOT learning during training")
                    print("  → This explains why outputs are uniform/random")
                    print("  → RayBNN needs to be trained separately first")
                elif changed_fraction < 0.1:
                    print(f"WARNING: Only {changed_fraction*100:.1f}% of parameters changed")
                    print("  → RayBNN is barely learning")
                    print("  → May need more training or different learning rate")
                elif mean_change < 1e-5:
                    print("WARNING: RayBNN parameters changed but very little")
                    print(f"  → Mean change: {mean_change:.6e} (very small)")
                    print("  → RayBNN learning is too slow")
                else:
                    print("RayBNN parameters ARE updating")
                    print(f"  → Mean change: {mean_change:.6e}")
                    print(f"  → {changed_fraction*100:.1f}% of parameters changed")
                    print("  → RayBNN is learning!")
                
                # Show parameter statistics
                print("\nParameter statistics:")
                print(f"  Before: mean={raybnn_params_before.mean():.6e}, std={raybnn_params_before.std():.6e}")
                print(f"  After:  mean={raybnn_params_after.mean():.6e}, std={raybnn_params_after.std():.6e}")
                
            else:
                print("Cannot compare as numpy arrays")
                print(f"  Before type: {type(raybnn_params_before)}")
                print(f"  After type: {type(raybnn_params_after)}")
                
                # Try object identity check
                if raybnn_params_before is network_params_obj_after:
                    print("  → Objects are the SAME (same memory location)")
                    print("  → RayBNN parameters likely NOT updated")
                elif raybnn_params_before == network_params_obj_after:
                    print("  → Objects have EQUAL values")
                    print("  → RayBNN parameters likely NOT updated")
                else:
                    print("  → Objects are DIFFERENT")
                    print("  → RayBNN parameters may have been updated")
                    print("  → But cannot quantify the change without numpy conversion")
                
        except Exception as e:
            print(f"Error checking RayBNN parameters: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Could not check RayBNN parameters (initial extraction failed)")
    
    print("="*60 + "\n")
    
    # Plot training loss
    if epoch_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses, label='Training Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss of model when freeze CNN and train RayBNN')
        # Set x-axis ticks to integer values (spacing of 1)
        max_epoch = len(epoch_losses) - 1
        plt.xticks(np.arange(0, max_epoch + 1, 1))
        final_loss = epoch_losses[-1]
        plt.text(0.02, 0.95, f'Final loss: {final_loss:.4f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.legend()
        plt.grid(True)
        plot_filename = "plot_freeze_cnn_train_raybnn.png"
        plt.savefig(plot_filename)
        print(f"Loss plot saved to {plot_filename}")
        
        # Also try to parse batch-level losses from log file if it exists
        log_file = "output_freeze_cnn_train_raybnn.txt"
        if os.path.exists(log_file):
            batch_losses = []
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        # Match format: "  Batch 0: loss=2.3026, acc=0.088"
                        match = re.search(r"loss=([\d\.]+)", line)
                        if match:
                            batch_losses.append(float(match.group(1)))
                
                if batch_losses:
                    plt.figure(figsize=(12, 6))
                    plt.plot(batch_losses, label='Batch Loss', alpha=0.6, linewidth=0.5)
                    plt.xlabel('Batch')
                    plt.ylabel('Cross Entropy Loss')
                    plt.title('Batch-Level Training Loss of of model when freeze CNN and train RayBNN')
                    final_batch_loss = batch_losses[-1]
                    plt.text(0.02, 0.95, f'Final batch loss: {final_batch_loss:.4f}',
                             transform=plt.gca().transAxes, fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                    plt.legend()
                    plt.grid(True)
                    batch_plot_filename = "cnn_batch_freeze_cnn_train_raybnn.png"
                    plt.savefig(batch_plot_filename)
                    print(f"Batch-level loss plot saved to {batch_plot_filename}")
            except Exception as e:
                print(f"Error parsing batch losses from log file: {e}")
    
    return model





if __name__ == '__main__':
    # Set this flag to run Test 7 or normal training
    
    
    end_to_end_model, x_train_tensor, y_train_tensor = main()

    trained_model = train_ete_model(end_to_end_model, x_train_tensor, y_train_tensor, batch_size=1000, max_epoch=5)
    print("Done without errors!")
    
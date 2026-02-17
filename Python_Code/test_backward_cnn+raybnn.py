


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

    alpha0 = 0.01 # 0.001

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

    max_epoch=7

    class CNN(nn.Module):
        def __init__(self):
            super (CNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, 
            kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=64, 
            kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=16,
            kernel_size=3, stride=1, padding=1)
            self.drop = nn.Dropout2d(p=0.1)  # changed to 0.1

            # # just added
            #self.projection = nn.Linear(784, 256)
            # self.bn = nn.BatchNorm1d(200)

        def forward(self, raw_images, y_labels, verbose=False):
            # First convolutional layer + pooling + ReLU
            x=self.conv1(raw_images)
            x=F.relu(x)
            x=self.pool(x)
            if verbose:
                print("After conv1 + pool + ReLU: ", x.shape) # torch.Size([1000, 128, 14, 14])

            # Second conv layer + pooling + ReLU
            x=self.conv2(x)
            x=F.relu(x)
            x=self.pool(x)
            if verbose:
                print("After conv2 + pool + ReLU: ", x.shape) # torch.Size([1000, 64, 7, 7])
            
            # Third conv layer + dropout + ReLU
            x=self.conv3(x)
            x=F.relu(x)
            if verbose:
                print("After conv3 + ReLU: ", x.shape) # torch.Size([1000, 16, 7, 7])



            # Dropout behavior depends on model.training:
            # - train(): randomly zeros ~10% of activations (regularization)
            # - eval(): no dropout, all activations pass through (consistent inference)
            features = F.dropout(x, training=self.training)
            if verbose:
                print("After dropout: ", features.shape) # torch.Size([1000, 24, 7, 7])

            features_flat = features.reshape(features.size(0), -1)
            if verbose:
                print("After flattening: ", features_flat.shape) # 16x7x7=784
            
            # # Apply projection
            # features_reduced = self.projection(features_flat)
            # if verbose:
            #     print("After projection: ", features_reduced.shape) # torch.Size([1000, 200])
            # features_normalized = self.bn(features_reduced) # Apply BN here
            return features_flat

    class AutoGradEndtoEnd(torch.autograd.Function):

        _current_epoch = 0
        @staticmethod
        def forward(ctx, features_flat, y_labels, batch_size, 
        traj_size, max_epoch, input_size, output_size, training_samples, alpha0):

            
            # Save tensors that will be needed in backward
            ctx.batch_size = batch_size
            ctx.traj_size = traj_size
            ctx.max_epoch = max_epoch
            ctx.input_size = input_size
            ctx.output_size = output_size
            ctx.training_samples = training_samples
            ctx.alpha0 = alpha0

            # print(f"[AUTOGRAD FORWARD] Features shape: {features_flat.shape}") # torch.Size([1000, 1176])
            # print(f"[AUTOGRAD FORWARD] Labels shape: {y_labels.shape}") # torch.Size([1000])

            # Convert X and Y to numpy arrays (Convert PyTorch → NumPy)
            features_np = features_flat.detach().cpu().numpy()
            y_labels_np = y_labels.detach().cpu().numpy()

            # Create training arrays using existing format (Reshape to RayBNN format)
            train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
            train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)

            print("train X shape: ", train_x.shape) # (256, 1000, 1, 60)
            # Divide raw dataset into correspond batches (Fill the arrays)
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
            
            # Check predictions
            preds = Yhat.argmax(dim=1)
            probs = torch.softmax(Yhat, dim=1)
            #print(f"probs: {probs}")
            # Sample outputs for inspection
            #print(f"\nSample outputs (first 5 samples):")
            for i in range(min(5, Yhat.shape[0])):
                pred_class = preds[i].item()
                confidence = probs[i, pred_class].item()
                true_label = y_labels[i].item() if i < y_labels.shape[0] else 'N/A'
                # print(f"  Sample {i}: predicted={pred_class}, "
                #       f"confidence={confidence:.3f}, "
                #       f"true_label={true_label}")
            
            print("="*50 + "\n")

            # Store entropy for epoch tracking
            probs = torch.softmax(Yhat, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            # Store in a way that can be accessed by the training loop
            ctx.entropy = entropy.item()
            
            return Yhat

        @staticmethod
        def backward(ctx, grad_output):
            features_flat, y_labels = ctx.saved_tensors
            batch_size = ctx.batch_size
            traj_size = ctx.traj_size
            max_epoch = ctx.max_epoch
            input_size = ctx.input_size
            output_size = ctx.output_size
            training_samples = ctx.training_samples
            alpha0 = ctx.alpha0

            current_epoch = AutoGradEndtoEnd._current_epoch

            # Convert PyTorch → NumPy
            features_np = features_flat.detach().cpu().numpy()
            y_labels_np = y_labels.detach().cpu().numpy()

            print("features_np shape: ", features_np.shape)
            print("y_label shape: ", y_labels_np.shape)

            # Create training arrays using the same format as forward pass (Reshape to RayBNN format)
            train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
            train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)

            # Format data the same way as forward pass (Fill the arrays)
            for i in range(features_np.shape[0]):
                j = (i % batch_size)
                k = int(i/batch_size)

                train_x[:, j, 0, k] = features_np[i, :]
                idx = int(y_labels_np[i])
                if idx < output_size:
                    train_y[idx, j, 0, k] = 1.0
            
            # Call RayBNN backward pass
            try:
                print(f"[AUTO-GRADIENT BACKWARD] Calling RayBNN backward with train_x shape: {train_x.shape}") # (1176, 1000, 1, 60)

                grad_result = raybnn_python.state_space_backward_group2(
                    train_x, train_y, traj_size, max_epoch, alpha0, arch_search, current_epoch
                )
                print("grad_result shape: ", grad_result.shape)

                # Convert grad_result to numpy array if needed (BEFORE using it)
                if not isinstance(grad_result, np.ndarray):
                    grad_result = np.array(grad_result, dtype=np.float32)

                grad_result_reshaped = grad_result[:, :, 0, 0].T     
                print("grad_result_reshaped shape: ", grad_result_reshaped.shape)
                
                # grad_result = grad_result * 1000.0  # Multiply by 1000x

                assert not np.isnan(grad_result).any(), "NaN in gradients!"

                grad_features = torch.from_numpy(grad_result_reshaped).to(features_flat.device)
                
                assert grad_features.shape == features_flat.shape, \
                f"Gradient shape {grad_features.shape} doesn't match features {features_flat.shape}"

                # === Print RayBNN Gradient before go back to CNN ===
                print("\n" + "="*70)
                print(f"DIAGNOSTIC: RayBNN Gradient → CNN (Epoch {current_epoch})")
                print("="*70)
                print(f"  Shape: {grad_features.shape}")
                print(f"  Mean: {grad_features.mean().item():.8f}")
                print(f"  Std: {grad_features.std().item():.8f}")
                print(f"  Max: {grad_features.max().item():.8f}")
                print(f"  Min: {grad_features.min().item():.8f}")
                # KEY CHECK: Is RayBNN using grad_output at all?
                # Compare magnitude of RayBNN gradient vs upstream gradient
                upstream_magnitude = grad_output.abs().mean().item()
                raybnn_grad_magnitude = grad_features.abs().mean().item()
                print(f"\n  Upstream grad magnitude (from loss): {upstream_magnitude:.8f}")
                print(f"  RayBNN grad magnitude (to CNN):      {raybnn_grad_magnitude:.8f}")
                print(f"  Ratio (RayBNN/upstream):              {raybnn_grad_magnitude / (upstream_magnitude + 1e-12):.4f}")
                
                # Check if gradient varies per sample (or is identical for all)
                grad_per_sample_norm = grad_features.norm(dim=1)
                print(f"\n  Per-sample gradient norm std: {grad_per_sample_norm.std().item():.8f}")
                if grad_per_sample_norm.std().item() < 1e-8:
                    print("  ✗ CRITICAL: All samples get identical gradients!")
                    print("    RayBNN backward is NOT differentiating between samples")
                else:
                    print("  ✓ Gradients vary across samples")
                
                # Check correlation between gradient and features
                correlation = torch.corrcoef(
                    torch.stack([grad_features.flatten(), features_flat.flatten()])
                )[0, 1].item()
                print(f"  Gradient-feature correlation: {correlation:.6f}")
                print("="*70 + "\n")

                print("\n Backward pass completed!")
            except Exception as e:
                print(f"[AUTOGRAD BACKWARD] Error calling RayBNN backward: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: return pass-through gradients
                grad_features = torch.zeros_like(features_flat)  # Create fallback gradient
                print("[AUTOGRAD BACKWARD] Using zero gradients as fallback")
            
            return grad_features, None, None, None, None, None, None, None, None

    class EndtoEndTrainer (nn.Module):
        def __init__(self, batch_size, traj_size, max_epoch, input_size, output_size, training_samples, alpha0):
            super().__init__()
            self.cnn = CNN()
            # Store RayBNN parameters for AutoGradEndtoEnd
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
            
            # Print CNN Output (Features) Before RayBNN ===
            if verbose:
                print("\n" + "="*70)
                print("DIAGNOSTIC: CNN Features BEFORE entering RayBNN")
                print("="*70)
                print(f"  Shape: {features.shape}")
                print(f"  Mean: {features.mean().item():.6f}")
                print(f"  Std: {features.std().item():.6f}")
                print(f"  Min: {features.min().item():.6f}")
                print(f"  Max: {features.max().item():.6f}")
                #print(f"  % zeros: {(features == 0).sum().item() / features.numel() * 100:.2f}%")
                print(f"  requires_grad: {features.requires_grad}")
                # print(f"  grad_fn: {features.grad_fn}")
                # Check if features are discriminative (different inputs → different features)
                feature_variance_per_sample = features.var(dim=0).mean().item()
                feature_variance_per_feature = features.var(dim=1).mean().item()
                print(f"  Variance across samples (per feature): {feature_variance_per_sample:.6f}")
                print(f"  Variance across features (per sample): {feature_variance_per_feature:.6f}")
                if feature_variance_per_sample < 1e-6:
                    print("  ✗ CRITICAL: All samples produce nearly identical features!")
                else:
                    print("  ✓ Features vary across samples")
                print("="*70 + "\n")
            output = AutoGradEndtoEnd.apply(
                features, # NORMALIZED CNN features for stability
                y_labels,          # labels
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


    end_to_end_model = EndtoEndTrainer( batch_size, traj_size, max_epoch, input_size, output_size, training_samples, alpha0)
    
    return end_to_end_model, x_train_tensor, y_train_tensor, alpha0


def train_ete_model(model, x_train, y_train, alpha0, batch_size, max_epoch, mode="both"):
    
    # ========== AREA 1 Print CNN Parameter Updates ==========
    print("\n" + "="*70)
    print("AREA 1 DIAGNOSTIC: Checking CNN Parameter Updates")
    print("="*70)
    
    # Store initial CNN parameters
    cnn_params_init = {
        name: param.clone().detach() 
        for name, param in model.cnn.named_parameters()
    }
    
    print("\n=== Initial CNN Parameters ===")
    for name, param in cnn_params_init.items():
        print(f"{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {param.mean().item():.6f}")
        print(f"  Std: {param.std().item():.6f}")
        print(f"  Min/Max: [{param.min().item():.6f}, {param.max().item():.6f}]")
    
    ## Mode Parameter
    # mode: "both" - Train CNN + RayBNN
    # mode: "cnn_only" - Train CNN, freeze RayBNN (alpha0=0)
    # mode: "raybnn_only" - Freeze CNN, train RayBNN only
    # mode: "frozen" - Freeze both (sanity check)
    if mode == "cnn_only":
        for param in model.cnn.parameters():
            param.requires_grad = True
        lr = 0.001
        alpha0 = 0.0 # Freeze RayBNN
        print(f"MODE: Train CNN only with CNN_lr={lr}, RayBNN_lr={alpha0}")

    elif mode =="raybnn_only":
        for param in model.cnn.parameters():
            param.requires_grad = False
        lr = 0.0
        print(f"MODE: Train RayBNN only with lr={alpha0}, CNN_lr={lr})")
    
    elif mode == "frozen":
        for param in model.cnn.parameters():
            param.requires_grad = False
        lr = 0.0
        alpha0 = 0.0
        print(f"MODE: Both frozen (lr={lr}, alpha0={alpha0})")

    else:
        for param in model.cnn.parameters():
            param.requires_grad = True
        lr = 0.001
        # alpha0 stays as passed in
        print(f"MODE: Train both (lr={lr}, alpha0={alpha0})")

    # Propagate alpha0 to model so backward() uses the mode-adjusted value
    model.alpha0 = alpha0

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 
        lr=lr
    ) if lr > 0 else None
    criterion = torch.nn.CrossEntropyLoss()
    
    # Check if CNN parameters are in optimizer
    if optimizer:
        print("\n=== CNN Parameters in Optimizer Check ===")
        optimizer_param_ids = set(id(p) for group in optimizer.param_groups for p in group['params'])
        cnn_param_ids = {name: id(param) for name, param in model.cnn.named_parameters()}
        
        for name, param_id in cnn_param_ids.items():
            in_optimizer = param_id in optimizer_param_ids
            status = "✓ YES" if in_optimizer else "✗ NO"
            print(f"{name}: {status}")
        
        all_in_optimizer = all(param_id in optimizer_param_ids for param_id in cnn_param_ids.values())
        if all_in_optimizer:
            print("\n✓ ALL CNN parameters are in optimizer")
        else:
            print("\n✗ WARNING: Some CNN parameters NOT in optimizer!")
    else:
        print("\n=== No optimizer (lr=0) — CNN parameters frozen ===")
    
    print("="*70 + "\n")
    
    print("x train shape: ",x_train.shape)  # torch.Size([60000, 1, 28, 28])
    print("y train shape: ", y_train.shape) # torch.Size([60000])
    
    # Set model to training mode
    model.train()
    print(f"\n{'='*50}")
    print(f" TRAINING: {max_epoch} epochs, batch_size={batch_size}")
    print(f" alpha0={alpha0}, lr={lr}")
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

            if optimizer:
                optimizer.zero_grad()

            # Reduce verbosity - only verbose for first batch of each epoch
            verbose = (i == 0 and epoch < 3)  # Only first 3 epochs, first batch
            output = model(batch_x, batch_y, verbose=verbose)
            
            
            
            # ========== AREA 4: Print RayBNN Output Analysis ==========
            if i == 0:  # First batch of each epoch
                print("\n" + "="*70)
                print(f"AREA 4 DIAGNOSTIC: Model Final Output Analysis (Epoch {epoch+1}, Batch {i})")
                print("="*70)
                
                with torch.no_grad():
                    print(f"\nOutput Statistics:")
                    print(f"  Shape: {output.shape}")
                    print(f"  Range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                    print(f"  Mean: {output.mean().item():.4f}")
                    print(f"  Std: {output.std().item():.4f}")
                    # print(f"  Variance: {output.var().item():.4f}")
                    
                    # Check if output is changing
                    if epoch == 0:
                        model._first_output = output.clone()
                        print(f"  Status: First epoch - storing baseline")
                    else:
                        output_change = (output - model._first_output).abs().mean().item()
                        print(f"  Output change from epoch 0: {output_change:.6f}")
                        if output_change < 1e-6:
                            print(f"  ✗ WARNING: Output NOT changing significantly!")
                        else:
                            print(f"  ✓ Output IS changing")
                    
                    # Prediction analysis
                    probs = torch.softmax(output, dim=1)
                    preds = output.argmax(dim=1)
                    
                    print(f"\nPrediction Analysis:")
                    print(f"  Predicted classes (first 20): {preds[:20].tolist()}")
                    print(f"  True labels (first 20): {batch_y[:20].tolist()}")
                    
                    # Confidence analysis
                    max_probs = probs.max(dim=1)[0]
                    print(f"\nConfidence Statistics:")
                    print(f"  Mean confidence: {max_probs.mean().item():.4f}")
                    print(f"  Min confidence: {max_probs.min().item():.4f}")
                    print(f"  Max confidence: {max_probs.max().item():.4f}")
                    
                    # Entropy
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
                    print(f"  Entropy: {entropy:.4f}/2.303 (random=2.303)")
                    
                    if entropy > 2.2:
                        print(f"  ✗ WARNING: High entropy - predictions are near-random!")
                    elif entropy < 1.5:
                        print(f"  ✓ EXCELLENT: Low entropy - confident predictions")
                    else:
                        print(f"  ✓ GOOD: Moderate entropy - learning in progress")
                
                print("="*70 + "\n")
            
            loss = criterion(output, batch_y)
            
            # CRITICAL FIX: Loss explosion detection
            if loss.item() > 10.0 or torch.isnan(loss) or torch.isinf(loss):
                print(f"LOSS EXPLOSION DETECTED: {loss.item():.3f}")
                print("Stopping training to prevent further damage")
                return model  # Early termination

            if mode in ("both", "cnn_only"):
                loss.backward()
            
            # ========== AREA 2: Print Gradient Analysis ==========
            if i == 0:  # First batch of each epoch
                print("\n" + "="*70)
                print(f"AREA 2 DIAGNOSTIC: CNN Parameters Gradients computed during backprop of (Epoch {epoch+1}, Batch {i})")
                print("="*70)
                
                print("\n=== CNN Gradients ===")
                for name, param in model.cnn.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        grad_max = param.grad.abs().max().item()
                        grad_min = param.grad.abs().min().item()
                        
                        print(f"\n{name}:")
                        print(f"  Gradient mean: {grad_mean:.8f}")
                        print(f"  Gradient max: {grad_max:.8f}")
                        print(f"  Gradient min: {grad_min:.8f}")
                        
                        # Diagnose gradient issues
                        if grad_mean < 1e-6:
                            print(f"  ✗ CRITICAL: Vanishing gradients! (mean < 1e-6)")                       
                        elif grad_mean > 1.0:
                            print(f"  ⚠ WARNING: Large gradients (mean > 1.0)")
                        else:
                            print(f"  ✓ Healthy gradient magnitude")
                        
                        # Check for all-zero gradients
                        if (param.grad == 0).all():
                            print(f"  ✗ CRITICAL: ALL gradients are zero!")
                    else:
                        print(f"\n{name}: ✗ NO GRADIENT (requires_grad might be False)")
                
                print("\n" + "="*70 + "\n")

            if optimizer:
                optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            epoch_total += batch_y.size(0)
            epoch_correct += (predicted == batch_y).sum().item()
            
            with torch.no_grad():
                probs = torch.softmax(output, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
                batch_entropies.append(entropy)
            # Progress indicator every 20 batches
            
            current_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            if i % 10 == 0 or i == 0:  # Print every 10 batches
                print(f"  Batch {i}/{batch_idx}: loss={loss.item():.4f}, acc={current_acc:.3f}, entropy={entropy:.3f}")

    return model


if __name__ == '__main__':    
    
    end_to_end_model, x_train_tensor, y_train_tensor , alpha0= main()

    trained_model = train_ete_model(end_to_end_model, x_train_tensor, 
    y_train_tensor,alpha0 , batch_size=1000, 
    max_epoch=7, mode="raybnn_only")
    print("Done without errors!")
    
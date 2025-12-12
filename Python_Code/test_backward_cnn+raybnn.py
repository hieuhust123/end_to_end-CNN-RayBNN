


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
    max_input_size = 256
    input_size = 256

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
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, 
            kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, 
            kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=16,
            kernel_size=3, stride=1, padding=1)
            self.drop = nn.Dropout2d(p=0.1)  # changed to 0.1

            # # just added
            self.projection = nn.Linear(784, 256)
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
            x=self.drop(x)


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
            features_reduced = self.projection(features_flat)
            if verbose:
                print("After projection: ", features_reduced.shape) # torch.Size([1000, 200])
            # features_normalized = self.bn(features_reduced) # Apply BN here
            return features_reduced

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
            print(f"probs: {probs}")
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
                grad_result_reshaped = grad_result[:, :, 0, 0].T     
                print("grad_result_reshaped shape: ", grad_result_reshaped.shape)

                # Convert grad_result to numpy array if needed
                if not isinstance(grad_result, np.ndarray):
                    grad_result = np.array(grad_result, dtype=np.float32)
                
                # grad_result = grad_result * 1000.0  # Multiply by 1000x

                assert not np.isnan(grad_result).any(), "NaN in gradients!"

                grad_features = torch.from_numpy(grad_result_reshaped).to(features_flat.device)
                
                assert grad_features.shape == features_flat.shape, \
                f"Gradient shape {grad_features.shape} doesn't match features {features_flat.shape}"

                print("\n Backward pass completed!")
            except Exception as e:
                print(f"[AUTOGRAD BACKWARD] Error calling RayBNN backward: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: return pass-through gradients
                grad_features = torch.zeros_like(features_flat)  # Create fallback gradient
                print("[AUTOGRAD BACKWARD] Using zero gradients as fallback")
            
            return grad_features, None,None,None,None,None,None, None, None, None

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


def train_ete_model(model, x_train, y_train,alpha0, batch_size, max_epoch):
    
    # ========== AREA 1 DIAGNOSTIC: CNN Parameter Updates ==========
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
    
    # Setup optimizer first to check parameter inclusion
    for param in model.cnn.parameters():
        param.requires_grad = True
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Check if CNN parameters are in optimizer
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
    
    print("="*70 + "\n")
    
    # Training tracking
    epoch_losses = []
    epoch_accuracies = []
    epoch_raybnn_entropies = []
    best_loss = float('inf')
    epoch_times = []  
    patience_counter = 0
    early_stop_patience = 5
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

            optimizer.zero_grad()

            # Reduce verbosity - only verbose for first batch of each epoch
            verbose = (i == 0 and epoch < 3)  # Only first 3 epochs, first batch
            output = model(batch_x, batch_y, verbose=verbose)
            
            # ========== AREA 4 DIAGNOSTIC: Output Analysis ==========
            if i == 0:  # First batch of each epoch
                print("\n" + "="*70)
                print(f"AREA 4 DIAGNOSTIC: RayBNN Output Analysis (Epoch {epoch+1}, Batch {i})")
                print("="*70)
                
                with torch.no_grad():
                    print(f"\nOutput Statistics:")
                    print(f"  Shape: {output.shape}")
                    print(f"  Range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                    print(f"  Mean: {output.mean().item():.4f}")
                    print(f"  Std: {output.std().item():.4f}")
                    print(f"  Variance: {output.var().item():.4f}")
                    
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
                    
                    # # Check prediction diversity
                    # unique_preds = torch.unique(preds)
                    # print(f"  Unique predictions: {len(unique_preds)}/10 classes")
                    # if len(unique_preds) < 5:
                    #     print(f"  ✗ WARNING: Model predicting only {len(unique_preds)} classes!")
                    # else:
                    #     print(f"  ✓ Good prediction diversity")
                    
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
            
            # CRITICAL FIX 7: Loss explosion detection
            if loss.item() > 10.0 or torch.isnan(loss) or torch.isinf(loss):
                print(f"LOSS EXPLOSION DETECTED: {loss.item():.3f}")
                print("Stopping training to prevent further damage")
                return model  # Early termination

            loss.backward()
            
            # ========== AREA 2 DIAGNOSTIC: Gradient Analysis ==========
            if i == 0:  # First batch of each epoch
                print("\n" + "="*70)
                print(f"AREA 2 DIAGNOSTIC: Gradient Flow Analysis (Epoch {epoch+1}, Batch {i})")
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
                        if grad_mean < 1e-10:
                            print(f"  ✗ CRITICAL: Vanishing gradients! (mean < 1e-10)")
                        elif grad_mean < 1e-6:
                            print(f"  ⚠ WARNING: Very small gradients (mean < 1e-6)")
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

        # ========== AREA 1 DIAGNOSTIC: Parameter Change Analysis ==========
        print("\n" + "="*70)
        print(f"AREA 1 DIAGNOSTIC: CNN Parameter Changes (End of Epoch {epoch+1})")
        print("="*70)
        
        total_change = 0.0
        max_change = 0.0
        min_change = float('inf')
        
        print("\n=== Parameter Updates ===")
        for name, param in model.cnn.named_parameters():
            param_init = cnn_params_init[name]
            change = (param - param_init).abs().mean().item()
            rel_change = change / (param_init.abs().mean().item() + 1e-10)
            
            total_change += change
            max_change = max(max_change, change)
            min_change = min(min_change, change)
            
            print(f"\n{name}:")
            print(f"  Absolute change: {change:.8f}")
            print(f"  Relative change: {rel_change:.8f}")
            print(f"  Current mean: {param.mean().item():.6f}")
            print(f"  Initial mean: {param_init.mean().item():.6f}")
            
            # Diagnose parameter update issues
            if change < 1e-10:
                print(f"  ✗ CRITICAL: Parameters NOT updating! (change < 1e-10)")
            elif change < 1e-6:
                print(f"  ⚠ WARNING: Very small updates (change < 1e-6)")
            else:
                print(f"  ✓ Parameters are updating")
        
        avg_change = total_change / len(cnn_params_init)
        print(f"\n=== Summary ===")
        print(f"Average parameter change: {avg_change:.8f}")
        print(f"Max parameter change: {max_change:.8f}")
        print(f"Min parameter change: {min_change:.8f}")
        
        if avg_change < 1e-8:
            print(f"\n✗ CRITICAL ISSUE: CNN parameters barely changing!")
            print(f"   Possible causes:")
            print(f"   1. Learning rate too small (current: {lr})")
            print(f"   2. Gradients vanishing (check AREA 2)")
            print(f"   3. RayBNN not passing gradients back")
        elif avg_change > 0.1:
            print(f"\n⚠ WARNING: Very large parameter changes (might be unstable)")
        else:
            print(f"\n✓ CNN parameters updating at healthy rate")
        
        print("="*70 + "\n")

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
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Enhanced Epoch Summary
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print(f"{'='*70}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Loss: {avg_loss:.6f} (change: {avg_loss - epoch_losses[-2] if len(epoch_losses) > 1 else 'N/A'})")
        print(f"  Accuracy: {epoch_accuracy:.4f} ({epoch_accuracy*100:.1f}%)")
        print(f"  RayBNN Entropy: {avg_entropy:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Alpha0 (RayBNN): {alpha0:.6f}")

        # Learning progress check
        if len(epoch_losses) > 1:
            loss_change = epoch_losses[-2] - avg_loss
            if abs(loss_change) < 1e-6:
                print(f"\n  ✗ LEARNING STATUS: STUCK (loss not changing)")
            elif loss_change > 0:
                print(f"\n  ✓ LEARNING STATUS: IMPROVING (loss decreased by {loss_change:.6f})")
            else:
                print(f"\n  ⚠ LEARNING STATUS: DEGRADING (loss increased by {-loss_change:.6f})")

        # RayBNN learning progress
        if avg_entropy < 2.0:
            print(f"  ✓ RayBNN showing learning progress! (entropy < 2.0)")
        elif avg_entropy < 2.2:
            print(f"  ✓ RayBNN showing some patterns (entropy < 2.2)")
        else:
            print(f"  ✗ RayBNN still exploring (high entropy)")
        
        print("="*70 + "\n")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered (no improvement for {early_stop_patience} epochs)")
            break
            
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
    
    print("="*60 + "\n")
    
    if alpha0 != 0 and lr !=0:
        mode = "cnn+raybnn_training"
        title_suffix = "Training Both CNN and RayBNN"
    elif alpha0 == 0 and current_lr != 0:
        mode = "train_cnn_freeze_raybnn"
        title_suffix = "Training CNN, Freezing RayBNN"
    elif alpha0 != 0 and current_lr == 0:
        mode = "freeze_cnn_train_raybnn"
        title_suffix = "Freezing CNN, Training RayBNN"
    else:  # both == 0
        mode = "freeze_both"
        title_suffix = "Freezing Both CNN and RayBNN"
    # Plot training loss
    if epoch_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses, label='Training Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title(f'Training Loss - {title_suffix}')
        # Set x-axis ticks to integer values (spacing of 1)
        max_epoch = len(epoch_losses) - 1
        plt.xticks(np.arange(0, max_epoch + 1, 1))
        final_loss = epoch_losses[-1]
        plt.text(0.02, 0.95, f'Final loss: {final_loss:.4f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.legend()
        plt.grid(True)
        plot_filename = f"plot_{mode}.png"
        plt.savefig(plot_filename)
        print(f"Loss plot saved to {plot_filename}")
        
        # Also try to parse batch-level losses from log file if it exists
        log_file = f"output_{mode}.txt"
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
                    plt.title(f'Batch-Level Training Loss - {title_suffix}')
                    final_batch_loss = batch_losses[-1]
                    plt.text(0.02, 0.95, f'Final batch loss: {final_batch_loss:.4f}',
                             transform=plt.gca().transAxes, fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                    plt.legend()
                    plt.grid(True)
                    batch_plot_filename = f"batch_{mode}.png"
                    plt.savefig(batch_plot_filename)
                    print(f"Batch-level loss plot saved to {batch_plot_filename}")
            except Exception as e:
                print(f"Error parsing batch losses from log file: {e}")
    
    return model


if __name__ == '__main__':    
    
    end_to_end_model, x_train_tensor, y_train_tensor , alpha0= main()

    trained_model = train_ete_model(end_to_end_model, x_train_tensor, y_train_tensor,alpha0 , batch_size=1000, max_epoch=7)
    print("Done without errors!")
    
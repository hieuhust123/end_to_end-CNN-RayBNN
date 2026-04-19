# 2. Architecture: CNN + RayBNN Design

[← Back to README](README.md)

## System Architecture

```
Input EEG Signal (batch, 2, 3200, 1)
         │ [2 channels, 3200 time steps, width=1]
         │
    ┌────▼────┐
    │   CNN    │  ← Extracts features, reduces dimensionality
    │ (12 Conv)│    Output: (batch, 256) feature vectors
    └────┬────┘
         │
    ┌────▼──────────┐
    │  RayBNN       │  ← Classifies features to 4 sleep stages
    │ (256 → 4)     │    Output: (batch, 4) logits
    └────┬──────────┘
         │
    ┌────▼────────────┐
    │  Softmax + Loss  │  ← Cross-entropy loss + backprop
    └─────────────────┘
```

---

## CNN_EEG: The Feature Extractor

### Design Rationale

The CNN is modeled after **Xuan Chen's Keras implementation** for consistency and reproducibility. Key characteristics:
- **Input**: (batch, 2, 3200, 1) — 2 EEG channels, 3200 time samples, width=1 for Conv2D
- **Output**: (batch, 256) — 256-dimensional feature vectors
- **Architecture**: 12 convolutional blocks with batch norm, ReLU, max pooling

### Layer-by-Layer Breakdown

```
Block     In→Out Channels   Kernel    Pooling     Output Shape
─────────────────────────────────────────────────────────────
Input                                              (batch, 2, 3200, 1)
  1       2 → 32           (3,1)      (2,1)      (batch,32,1600,1)   ~195 MB @ batch=200
  2       32 → 64          (3,1)      (2,1)      (batch,64,800,1)    ~39 MB
  3       64 → 128         (3,1)      (2,1)      (batch,128,400,1)   ~39 MB
  4       128 → 128        (3,1)      (2,1)      (batch,128,200,1)   ~19.5 MB
  5       128 → 128        (3,1)      (2,1)      (batch,128,100,1)   (cached↓)
  6       128 → 128        (3,1)      (2,1)      (batch,128,50,1)
  7       128 → 256        (3,1)      (2,1)      (batch,256,25,1)
  8       256 → 256        (3,1)      (2,1)      (batch,256,12,1)    Small activations
  9       256 → 256        (3,1)      (2,1)      (batch,256,6,1)     No checkpointing
  10      256 → 256        (3,1)      (2,1)      (batch,256,3,1)
  11      256 → 256        (3,1)      (2,1)      (batch,256,2,1)
  12      256 → 256        (3,1)      (2,1)      (batch,256,1,1)
AdaptivePool                                       (batch,256,1,1)
Flatten                                            (batch,256)
Dense Head:  BN → Dropout(0.5) → FC(256→256) → ReLU → BN → Dropout(0.5)
Output                                             (batch,256)
```

### Key Implementation Details

**Kernel & Pooling Strategy**:
- Kernel `(3,1)` slides only along the time axis (output_time = input_time with padding='same')
- Pooling `(2,1)` halves the time dimension; width dimension stays 1
- After 12 blocks of (3,1) convs + (2,1) pooling: 
  - Time: 3200 → 1600 → 800 → 400 → 200 → 100 → 50 → 25 → 12 → 6 → 3 → 2 → 1

**Batch Norm + ReLU**:
```python
def _blockN(self, x):
    return self._pool(F.relu(self.bnN(self.convN(x))))
```
Each block applies: Conv → BatchNorm → ReLU → MaxPool

**Gaussian Noise Augmentation** (training only):
```python
if self.training:
    x = x + torch.randn_like(x) * self.noise_std  # noise_std = 0.0005
```

**Dense Head** (after flatten):
```python
x = BN(x)                    # Normalize 256-dim features
x = Dropout(0.5, x)         # Spatial dropout: drop 50% of activations
x = ReLU(FC(256→256, x))    # Dense layer + ReLU
x = BN(x)                    # Normalize again
x = Dropout(0.5, x)         # Another dropout
return x                     # (batch, 256) → fed to RayBNN
```

---

## RayBNN: The Probabilistic Classifier

### What Is RayBNN?

**RayBNN** (Ray Bayesian Neural Network) is an external Rust library that provides uncertainty-aware classification. We treat it as a **black box** in PyTorch.

**Key interface**:
```python
# Initialize
arch_search = raybnn_python.create_start_archtecture(
    input_size=256,     # Matches CNN output
    output_size=4,      # 4 sleep stages
    batch_size=1000,    # Fixed batch size
    ...
)

# Forward pass (inference + training)
Yhat = raybnn_python.state_space_forward_batch(
    train_x,     # (256, batch, traj_steps, 1) CNN features reshaped
    train_y,     # (4, batch, traj_size, 1) one-hot labels
    traj_size,   # Number of trajectory slots
    max_epoch,   # Current epoch (used inside)
    arch_search  # Model state
)
# Returns Yhat: (4, batch, traj_size, 1) logits

# Backward pass (gradient computation)
grad_result, raw_bnn_grad, params, _ = raybnn_python.state_space_backward_group2(
    train_x, train_y, traj_size, max_epoch, alpha0, arch_search, current_epoch
)
# Returns:
#   grad_result: (256, batch, traj_steps, 1) gradients w.r.t. CNN features
#   raw_bnn_grad: negated RayBNN gradient (for deferred update)
#   params: current RayBNN parameters
```

### Why Separate RayBNN?

- **Bayesian inference** is computationally expensive (marginalization over weight distributions)
- **Rust implementation** provides speed & numerical stability
- **Python bindings** allow seamless integration with PyTorch

---

## Integration: AutoGrad Function

### The Problem

PyTorch autodiff doesn't understand the Rust RayBNN code. We need a **custom autograd function** to bridge them.

### The Solution: `AutoGradEndtoEnd`

```python
class AutoGradEndtoEnd(torch.autograd.Function):
    """Custom autograd bridging CNN → RayBNN → Gradient backflow"""
    
    @staticmethod
    def forward(ctx, features_flat, y_labels, batch_size, traj_size, ...):
        # CPU conversion
        features_np = features_flat.detach().cpu().numpy()  # (T*batch, 256)
        y_labels_np = y_labels.detach().cpu().numpy()       # (T*batch,)
        
        # Reshape for RayBNN (expects 4D with trajectory dim)
        # traj_steps = traj_size + proc_num - 1 (warm-up slots)
        train_x = np.zeros((input_size, batch_size, traj_steps, 1), dtype=np.float32)
        train_y = np.zeros((output_size, batch_size, traj_size, 1), dtype=np.float32)
        
        # Fill real data slots; warm-up slots are zero-padded
        # ...reshaping logic...
        
        # Call Rust forward
        Yhat_array = raybnn_python.state_space_forward_batch(
            train_x, train_y, traj_size, max_epoch, AutoGradEndtoEnd._arch_search
        )
        
        # Convert back to PyTorch
        Yhat_tensor = torch.from_numpy(Yhat_array).to(features_flat.device)
        
        # Average over trajectory slots (since all slots have real labels)
        Yhat = Yhat_tensor.mean(dim=2).squeeze(-1).T  # (batch, 4)
        
        # Cache for backward
        ctx.save_for_backward(features_flat, y_labels)
        ctx.train_x = train_x
        ctx.train_y = train_y
        
        return Yhat  # (batch, 4) logits → loss computation
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve cached tensors
        features_flat, y_labels = ctx.saved_tensors
        train_x = ctx.train_x
        train_y = ctx.train_y
        
        # Call Rust backward
        grad_result, raw_bnn_grad, params, _ = raybnn_python.state_space_backward_group2(
            train_x, train_y, traj_size, max_epoch, alpha0, 
            AutoGradEndtoEnd._arch_search, current_epoch
        )
        
        # Extract gradients for real slots (discard warm-up)
        grad_real = grad_result[:, :, :traj_size, 0]  # (256, batch, T)
        grad_result_reshaped = grad_real.transpose(2,1,0).reshape(T*batch, 256)
        
        # Convert to PyTorch
        grad_features = torch.from_numpy(grad_result_reshaped).to(features_flat.device)
        
        # Deferred BNN update: STORE for later (after CNN optimizer.step)
        AutoGradEndtoEnd._pending_bnn_grad = raw_bnn_grad.flatten()
        AutoGradEndtoEnd._pending_bnn_params = params.flatten()
        
        return grad_features, None, None, ...  # Return gradient for features_flat
```

### Data Flow Through AutoGrad

```
PyTorch Forward:
  (batch, 2, 3200, 1) 
    ↓ [CNN.forward()]
  (batch, 256) features
    ↓ [AutoGradEndtoEnd.apply()]
  (batch, 4) Yhat logits
    ↓ [CrossEntropyLoss()]
  scalar loss
    ↓ [loss.backward()]

PyTorch Backward:
  grad_output: (batch, 4)
    ↓ [inside AutoGrad.backward()]
  Call Rust backward on train_x, train_y
    ↓ [Rust computes grad_result: (256, batch, traj_steps, 1)]
  Extract real slots + reshape
    ↓ grad_features: (batch, 256)
  Store RayBNN grad for later
    ↓ [return grad_features to CNN] 
  
  CNN backward (via PyTorch):
  grad_features: (batch, 256)
    ↓ [through CNN layers]
  CNN param gradients ready
    ↓ [collected into grad buffers]

Optimizer Step:
  CNN: optimizer.step()  ← Updates CNN from collected gradients
  RayBNN: apply_deferred_bnn_update()  ← Updates BNN from stored gradient
```

---

## End-to-End Trainer Class

```python
class EndtoEndTrainer(nn.Module):
    def __init__(self, batch_size, traj_size, max_epoch, ...):
        self.cnn = CNN_EEG()
        self.batch_size = batch_size
        self.traj_size = traj_size
        # (other hyperparams)
    
    def forward(self, raw_eeg, y_labels, verbose=True):
        # Flatten trajectory dimension into batch for CNN
        if traj_size > 1:
            raw_eeg = raw_eeg.permute(1,0,2,3,4).reshape(T*batch, 2, 3200, 1)
            y_labels = y_labels.permute(1,0).reshape(T*batch)
        
        # CNN forward with mixed precision (float16 on GPU for speed)
        with torch.amp.autocast(...):
            features = self.cnn(raw_eeg, y_labels, verbose)
        
        # Call custom autograd function
        output = AutoGradEndtoEnd.apply(
            features, y_labels, self.batch_size, self.traj_size, ...
        )
        
        return output  # (batch, 4) logits
```

---

## Gradient Checkpointing: Memory Efficiency

### The Problem
Blocks 1-2 have huge activations:
- Block 1 output: (200, 32, 1600, 1) ≈ 195 MB per batch
- Block 2 output: (200, 64, 800, 1) ≈ 39 MB
- **Total early-block VRAM**: ~95% of all allocations

Default PyTorch caches activations for backward pass → runs out of memory on modest GPUs.

### The Solution: Gradient Checkpointing

Instead of caching, **recompute activations during backward**:
- Forward: Compute block output, discard immediately (keep only input)
- Backward: Recompute forward to get activations, then compute gradients
- Trade: +30% compute time, -50% peak memory

```python
def forward_with_checkpointing(self, x):
    # Used for blocks 1-7 (large activations)
    x = torch.utils.checkpoint.checkpoint(self._block1, x, use_reentrant=False)
    # ...
    x = torch.utils.checkpoint.checkpoint(self._block7, x, use_reentrant=False)
    
    # Blocks 8-12: small enough to not checkpoint
    x = self._block8(x)  # No checkpoint
    # ...
    return x
```

---

## Memory Breakdown (batch_size=1000)

| Component | Memory |
|-----------|--------|
| CNN weights | ~8 MB |
| RayBNN weights | ~2 MB |
| Input batch (2,3200,1) × 1000 | ~25 MB |
| Block 1 activation (32,1600,1) × 1000 | ~195 MB (checkpointed) |
| Block 2 activation (64,800,1) × 1000 | ~39 MB (checkpointed) |
| Blocks 3-12 activations (progressive) | ~600 MB (checkpointed) |
| Optimizer state (Adam mt, vt) | ~20 MB |
| Gradients | ~50 MB |
| **Total w/ checkpointing** | **~1-2 GB** |
| **Total w/o checkpointing** | **~6-8 GB** |

Enables training on consumer GPUs (12 GB VRAM) that would otherwise fail.

---

## Next Steps

- **Data handling**: [Data Pipeline](03_DATA_PIPELINE.md)
- **Implementation details**: [Model Components](04_MODEL_COMPONENTS.md)
- **Training mechanics**: [Training Flow](05_TRAINING_FLOW.md)

---

[← Back to README](README.md) | [Previous: Overview](01_OVERVIEW.md) | [Next: Data Pipeline →](03_DATA_PIPELINE.md)

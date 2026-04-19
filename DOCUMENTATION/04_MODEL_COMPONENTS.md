# 4. Model Components: Implementation Deep Dive

[← Back to README](README.md)

## CNN_EEG: Layer Implementation

### Initialization

```python
class CNN_EEG(nn.Module):
    def __init__(self):
        super(CNN_EEG, self).__init__()
        
        # Gaussian noise parameters (training only)
        self.noise_std = 0.0005
        
        # Conv Blocks: 2→32→64→128×4→256×6
        self.conv1  = nn.Conv2d(2,   32,  kernel_size=(3, 1), padding=(1, 0))
        self.bn1    = nn.BatchNorm2d(32)
        
        self.conv2  = nn.Conv2d(32,  64,  kernel_size=(3, 1), padding=(1, 0))
        self.bn2    = nn.BatchNorm2d(64)
        
        # Blocks 3-6: 64→128 (x4)
        self.conv3  = nn.Conv2d(64,  128, kernel_size=(3, 1), padding=(1, 0))
        self.bn3    = nn.BatchNorm2d(128)
        # ... conv4-6, bn4-6 similar ...
        
        # Blocks 7-12: 128→256 (x6)
        self.conv7  = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0))
        self.bn7    = nn.BatchNorm2d(256)
        # ... conv8-12, bn8-12 similar ...
        
        # Pooling and adaptive pooling
        self._max_pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense head
        self.bn_dense1  = nn.BatchNorm1d(256)
        self.drop1      = nn.Dropout(p=0.5)
        self.fc1        = nn.Linear(256, 256)
        self.bn_dense2  = nn.BatchNorm1d(256)
        self.drop2      = nn.Dropout(p=0.5)
        
        # Weight initialization (Xavier uniform)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization: stabilizes early training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
```

### Kernel & Padding Explanation

**Kernel (3, 1)**:
- (3, 1) means: 3 taps along time axis, 1 tap along width axis
- Only captures temporal patterns (width=1 stays 1)

**Padding (1, 0)**:
- Pad = 1 along time axis: output_time = input_time (called "same" padding)
- Pad = 0 along width axis: output_width = input_width (always 1)

**Conv math**:
```
output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
            = floor((3200 + 2*1 - 3) / 1) + 1
            = floor(3200) + 1
            = 3200  ✓
```

### Pooling Strategy

```python
def _pool(self, x):
    """MaxPool(2,1) along time axis — skip when already at dimension 1"""
    if x.shape[2] > 1:  # Check time dimension (dim=2)
        return self._max_pool(x)
    return x
```

**Why check?** PyTorch's MaxPool2d errors if output would be < 1. After ~12 pooling ops, time dimension reaches 1 and further pooling is invalid. This guard replicates Keras behavior (which never errors).

### Block Forward Pass

```python
def _block1(self, x):
    # Input: (batch, 2, 3200, 1)
    x = self.conv1(x)        # (batch, 32, 3200, 1)
    x = self.bn1(x)          # Normalize across (batch, 3200, 1) dims
    x = F.relu(x)            # ReLU: max(0, x)
    x = self._pool(x)        # MaxPool(2,1) → (batch, 32, 1600, 1)
    return x

def _block2(self, x):
    # Input: (batch, 32, 1600, 1)
    x = self.conv2(x)        # (batch, 64, 1600, 1)
    x = self.bn2(x)
    x = F.relu(x)
    x = self._pool(x)        # → (batch, 64, 800, 1)
    return x

# ... similarly for blocks 3-12 ...
```

### Full Forward Pass

```python
def forward(self, x, y_labels, verbose=False):
    """
    Args:
        x: (batch, 2, 3200, 1) — EEG windows
        y_labels: (batch,) — not used in forward (API compatibility)
        verbose: Print shape diagnostics (usually False except diagnostics)
    
    Returns:
        (batch, 256) — feature vectors
    """
    
    # Gaussian noise augmentation (training only)
    if self.training:
        x = x + torch.randn_like(x) * self.noise_std
    
    # Decide whether to use checkpointing
    use_ckpt = self.training  # Only checkpoint during training
    
    # Blocks 1-7: large activations → use gradient checkpointing
    x = torch.utils.checkpoint.checkpoint(self._block1, x, use_reentrant=False) if use_ckpt else self._block1(x)
    if verbose: print(f"After block1 (32 filt): {x.shape}")  # (batch,32,1600,1)
    
    x = torch.utils.checkpoint.checkpoint(self._block2, x, use_reentrant=False) if use_ckpt else self._block2(x)
    if verbose: print(f"After block2 (64 filt): {x.shape}")  # (batch,64,800,1)
    
    # ... blocks 3-7 similarly ...
    
    # Blocks 8-12: small activations → NO checkpointing
    x = self._pool(F.relu(self.bn8(self.conv8(x))))
    if verbose: print(f"After block8  (256 filt): {x.shape}")
    
    x = self._pool(F.relu(self.bn9(self.conv9(x))))
    # ... blocks 10-12 similarly ...
    
    # Spatial collapse: (batch, 256, ?, 1) → (batch, 256, 1, 1)
    x = self.adaptive_pool(x)
    if verbose: print(f"After adaptive pool: {x.shape}")
    
    # Flatten: (batch, 256, 1, 1) → (batch, 256)
    x = x.reshape(x.size(0), -1)
    if verbose: print(f"After flatten: {x.shape}")
    
    # Dense head: BN → Dropout → FC → ReLU → BN → Dropout
    x = self.bn_dense1(x)          # Normalize features
    x = self.drop1(x)              # Spatial dropout (50%)
    x = F.relu(self.fc1(x))        # Dense(256→256) + ReLU
    x = self.bn_dense2(x)          # Normalize again
    x = self.drop2(x)              # Dropout (50%)
    
    # Output: (batch, 256)
    if verbose: print(f"After dense head (features): {x.shape}")
    return x
```

---

## Custom AutoGrad: Detailed Flow

### Forward Pass Deep Dive

```python
@staticmethod
def forward(ctx, features_flat, y_labels, batch_size, traj_size, 
            max_epoch, input_size, output_size, training_samples, alpha0):
    """
    Args:
        features_flat: (T*batch, 256) from CNN ← Already merged trajectory + batch dims
        y_labels: (T*batch,) class indices {0, 1, 2, 3}
        * other params: Hyperparameters needed in backward
    
    Returns:
        Yhat: (batch, 4) logits
    """
    
    # ──────── STEP 1: Convert PyTorch → NumPy (CPU) ────────
    # RayBNN Rust code only accepts NumPy arrays
    features_np = features_flat.detach().cpu().numpy()  # (T*batch, 256)
    y_labels_np = y_labels.detach().cpu().numpy()       # (T*batch,)
    
    # ──────── STEP 2: Reshape for RayBNN trajectory format ────────
    # RayBNN expects 4D tensors with explicit trajectory dimension
    traj_steps = traj_size + proc_num - 1  # proc_num typically 2
    # Example: traj_size=2, proc_num=2 → traj_steps=3
    # Why? RayBNN internally uses 3 time steps: 2 real + 1 warm-up
    
    train_x = np.zeros((input_size, batch_size, traj_steps, 1), dtype=np.float32)
    train_y = np.zeros((output_size, batch_size, traj_size, 1), dtype=np.float32)
    
    # Example shapes:
    # train_x: (256, 1000, 3, 1) ← 256 features, 1000 batch, 3 time steps, width 1
    # train_y: (4, 1000, 2, 1) ← 4 classes, 1000 batch, 2 real time steps, width 1
    
    # ──────── STEP 3: Fill train_x and train_y ────────
    # Reshape features from flat to trajectory layout
    features_traj = features_np.reshape(traj_size, batch_size, input_size)
    # Example: (T*batch, 256) → (2, 1000, 256)
    
    labels_traj = y_labels_np.reshape(traj_size, batch_size)
    # Example: (T*batch,) → (2, 1000)
    
    for t in range(traj_size):
        # Transpose features so they're in (256, 1000) layout
        train_x[:, :, t, 0] = features_traj[t].T  # (256, batch)
        
        # Convert class indices to one-hot format
        indices_t = labels_traj[t].astype(int)  # (1000,) with values {0,1,2,3}
        valid_t   = (indices_t >= 0) & (indices_t < output_size)  # Filter invalid labels
        
        # Set one-hot: y[class_idx, sample_idx, time_step, 0] = 1.0
        train_y[indices_t[valid_t], np.where(valid_t)[0], t, 0] = 1.0
    
    # traj_steps - traj_size warm-up slots remain zero (already initialized)
    
    # ──────── STEP 4: Call Rust forward ────────
    Yhat_array = raybnn_python.state_space_forward_batch(
        train_x,    # (256, 1000, 3, 1)
        train_y,    # (4, 1000, 2, 1)
        traj_size,  # 2
        max_epoch,  # Current epoch
        AutoGradEndtoEnd._arch_search  # Model state (persisted)
    )
    # Returns Yhat_array: (4, 1000, 2, 1) logits for each class+sample+time+width
    
    # ──────── STEP 5: Convert back to PyTorch + aggregate ────────
    Yhat_array = np.array(Yhat_array).astype(np.float32)
    Yhat_tensor = torch.from_numpy(Yhat_array).to(features_flat.device)
    # Yhat_tensor: (4, 1000, 2, 1)
    
    # Average predictions over traj_size dimension
    # (all trajectory slots have real labels, so pool them)
    Yhat = Yhat_tensor.mean(dim=2).squeeze(-1).T
    # (4, 1000, 2, 1) → mean(dim=2) → (4, 1000, 1)
    #                 → squeeze(-1) → (4, 1000)
    #                 → T → (1000, 4) ✓
    
    # ──────── STEP 6: Cache for backward ────────
    ctx.save_for_backward(features_flat, y_labels)
    ctx.train_x = train_x  # Cache formatted arrays
    ctx.train_y = train_y
    ctx.batch_size = batch_size
    ctx.traj_size = traj_size
    # ... (other hyperparams) ...
    
    return Yhat  # (batch, 4) logits
```

### Backward Pass Deep Dive

```python
@staticmethod
def backward(ctx, grad_output):
    """
    Args:
        grad_output: (batch, 4) — gradient from loss w.r.t. logits
                     Already computed by PyTorch's CrossEntropyLoss
    
    Returns:
        Gradients for all forward() inputs:
        (grad_features_flat, None, None, ...) 
        grad_features_flat: (T*batch, 256) ← backprop to CNN
    """
    
    # ──────── STEP 1: Retrieve cached data ────────
    features_flat, y_labels = ctx.saved_tensors
    train_x = ctx.train_x
    train_y = ctx.train_y
    batch_size = ctx.batch_size
    traj_size = ctx.traj_size
    max_epoch = ctx.max_epoch
    alpha0 = ctx.alpha0
    
    # ──────── STEP 2: Call Rust backward ────────
    # RayBNN computes gradients w.r.t. input features (train_x)
    grad_result, raw_bnn_grad, current_params, _ = \
        raybnn_python.state_space_backward_group2(
            train_x, train_y, traj_size, max_epoch, alpha0,
            AutoGradEndtoEnd._arch_search, current_epoch
        )
    # Returns:
    #   grad_result: (256, 1000, 3, 1) gradients w.r.t. train_x
    #               Slots 0..traj_size-1 = real gradients
    #               Slots traj_size..traj_steps-1 = warm-up (discard)
    #   raw_bnn_grad: (large vector) negated BNN gradients (for deferred update)
    #   current_params: Current BNN parameters
    
    # ──────── STEP 3: Extract real slots + reshape ────────
    # Discard warm-up slots, keep only traj_size real gradients
    grad_real = grad_result[:, :, :traj_size, 0]  # (256, batch, traj_size)
    
    # Reshape back to match flat input layout
    # (256, 1000, 2) → transpose to (2, 1000, 256) → reshape to (2*1000, 256)
    grad_result_reshaped = grad_real.transpose(2, 1, 0).reshape(traj_size * batch_size, 256)
    
    # ──────── STEP 4: Convert to PyTorch ────────
    grad_features = torch.from_numpy(grad_result_reshaped).to(features_flat.device)
    
    # Sanity check
    assert grad_features.shape == features_flat.shape, \
        f"Shape mismatch: {grad_features.shape} vs {features_flat.shape}"
    
    # ──────── STEP 5: Deferred BNN update ────────
    # Cache gradients for later (after CNN optimizer.step)
    # Reason: Both CNN and BNN should update based on same forward state
    AutoGradEndtoEnd._pending_bnn_grad = np.array(raw_bnn_grad, dtype=np.float32).flatten()
    AutoGradEndtoEnd._pending_bnn_params = np.array(current_params, dtype=np.float32).flatten()
    
    # ──────── STEP 6: Diagnostic logging ────────
    if AutoGradEndtoEnd._current_batch == 0:  # Only first batch of epoch
        print(f"[BACKWARD] dL/dX min={grad_features.min():.6f} "
              f"max={grad_features.max():.6f} mean={grad_features.mean():.6f}")
    
    # ──────── STEP 7: Return gradients ────────
    # Must return gradient for EVERY input to forward()
    # Most (batch_size, traj_size, ...) are None because they don't need gradients
    return grad_features, None, None, None, None, None, None, None, None
    #      └─ go to features_flat
    #         gradient w.r.t. CNN features (the only thing that matters!)
```

### Deferred BNN Update

```python
def apply_deferred_bnn_update(alpha0, autograd_cls):
    """
    Apply BNN weight update AFTER CNN optimizer.step().
    
    Problem: RayBNN backward computes gradients but doesn't update weights.
    Solution: Apply update here with Adam optimizer (matching Rust gd_f32.rs).
    """
    
    # Retrieve cached gradient from backward pass
    bnn_grad = autograd_cls._pending_bnn_grad  # (N,) flattened
    bnn_params = autograd_cls._pending_bnn_params  # (N,) flattened
    
    if bnn_grad is None:
        return  # No pending update
    
    # Initialize or load Adam state
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    
    if autograd_cls._adam_mt is None:
        autograd_cls._adam_mt = np.zeros_like(bnn_grad)  # First moment (mean)
        autograd_cls._adam_vt = np.zeros_like(bnn_grad)  # Second moment (variance)
    
    mt, vt = autograd_cls._adam_mt, autograd_cls._adam_vt
    
    # Adam update (vectorized)
    g = bnn_grad
    mt[:] = beta1 * mt + (1.0 - beta1) * g    # Running mean: m_t
    vt[:] = beta2 * vt + (1.0 - beta2) * g*g  # Running variance: v_t
    
    # Bias correction (important for early iterations)
    nmt = mt / (1.0 - beta1)  # Corrected mean
    nvt = np.sqrt(vt / (1.0 - beta2)) + epsilon  # Corrected variance
    
    # Weight update: θ_new = θ_old + α * m̂ / (√v̂ + ε)
    # NOTE: gradient is already negated from Rust, so we ADD
    bnn_params[:] += alpha0 * (nmt / nvt)
    
    # Persist updates back to RayBNN state
    autograd_cls._arch_search["neural_network"]["network_params"]["data"] = bnn_params.tolist()
    autograd_cls._adam_mt = mt
    autograd_cls._adam_vt = vt
    
    # Clear pending update
    autograd_cls._pending_bnn_grad = None
    autograd_cls._pending_bnn_params = None
```

---

## Checkpoint Mechanism

### How Gradient Checkpointing Works

```python
# Forward PASS without checkpointing (normal):
def forward_normal(self, x):
    x = self._block1(x)  # Cache activations from _block1
    x = self._block2(x)  # Cache activations from _block2
    return x
# Memory: ~50 MB (cached activations)

# Forward PASS with checkpointing:
def forward_checkpointed(self, x):
    x = torch.utils.checkpoint.checkpoint(self._block1, x)
    x = torch.utils.checkpoint.checkpoint(self._block2, x)
    return x
# Memory: ~2 MB (only store input to each block)

# Backward PASS with checkpointing:
# 1. Recompute _block1 forward pass (use saved input)
# 2. Get activations needed for gradient computation
# 3. Compute gradients, discard activations
# 4. Repeat for _block2
# Time trade-off: ~30% extra computation, but GPU memory halved
```

### When to Checkpoint

```python
# Blocks 1-7: Heavy activations
x = torch.checkpoint.checkpoint(self._block1, x, use_reentrant=False) if training else self._block1(x)
# → Checkpoint ONLY during training (inference doesn't need gradients)

# Blocks 8-12: Small activations (< 12 MB each)
x = self._block8(x)  # No checkpoint; too cheap to justify recompute
```

---

## Gradient Flow Diagram

```
Loss scalar
    ↑
    │ ∂loss/∂Yhat (batch, 4)
    │
[Softmax + CE Loss]  ← Computed by PyTorch
    ↑
    │ ∂loss/∂logits (batch, 4)
    │
[AutoGradEndtoEnd.backward()]  ← Custom
    │
    ├→ Call Rust: state_space_backward_group2(train_x, train_y)
    │  Returns: ∂loss/∂train_x (256, batch, traj_steps, 1)
    │
    ├→ Extract real slots, reshape to (T*batch, 256)
    │  ∂loss/∂features (T*batch, 256)
    │
    └→ Return to PyTorch backward graph
       
[CNN.backward()] ← PyTorch's autograd
    │ ∂loss/∂features (T*batch, 256)
    │
    ├→ [FC layer backward] → ∂loss/∂(after_dense_pool)
    ├→ [Dense pool backward] → ∂loss/∂(after_block12)
    ├→ ...
    └→ [Conv1 + BN1 backward] → ∂loss/∂(CNN params)
    
CNN Param Gradients:
    ├→ conv1.weight.grad, conv1.bias.grad
    ├→ bn1.weight.grad, bn1.bias.grad
    └→ ... (all 12 layers)

[Optimizer.step()]
    │ Updates CNN params using collected gradients
    │
[apply_deferred_bnn_update()]
    │ Updates BNN params using cached Rust gradient
```

---

## Next Steps

- **How training loop works**: [Training Flow](05_TRAINING_FLOW.md)
- **Running & debugging**: [Training Flow](05_TRAINING_FLOW.md)
- **Running the script**: [Running the Script](07_RUNNING_THE_SCRIPT.md)

---

[← Back to README](README.md) | [Previous: Data Pipeline](03_DATA_PIPELINE.md) | [Next: Training Flow →](05_TRAINING_FLOW.md)

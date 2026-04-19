# 3. Data Pipeline: Handling EEG Data

[← Back to README](README.md)

## Overview: EEG Data Lifecycle

```
MATLAB Files (.mat)
    │ [Raw EEG: 483,200 samples/subject @ 200 Hz ≈ 40 hours]
    │
    ├─→ [Load + Preprocess] (MWTDataset.__init__)
    │   • Normalize: clip to [-1,1]
    │   • Pad: add 1600 samples on each side (for windowing without edge effects)
    │   • Store in RAM: (483200,) → (2 channels × 76 subjects ≈ 280 MB total)
    │
    ├─→ [Train/Test Split] (make_4fold_splits)
    │   • Shuffle 76 subjects with fixed seed
    │   • Divide into 4 groups of ~19 subjects each
    │   • Fold k: train=57 subjects, test=19 subjects
    │
    └─→ [Batching via __getitem__] (MWTDataset.__getitem__)
        • On-demand: extract 16-second windows
        • Create trajectories (overlapping windows, 4s apart)
        • Output: (batch, T, 2, 3200, 1) + (batch, T) labels
```

---

## MWTDataset: Custom PyTorch Dataset

### Design Philosophy

**Memory efficiency**: Store only raw padded signals (~280 MB), generate windows on-the-fly during training.

```python
class MWTDataset(torch.utils.data.Dataset):
    def __init__(self, mat_dir, file_list, w_len=1600, stride=1, 
                 traj_size=1, traj_stride=800):
        """
        Args:
            mat_dir: Path to .mat files (e.g., '../mwt_eeg/')
            file_list: List of .mat filenames to load (e.g., ['subj_000.mat', ...])
            w_len: Half-window size = 1600 samples ← 8 seconds at 200 Hz
                   Full window = 2 * w_len = 3200 samples = 16 seconds
            stride: Sampling interval (stride=50 → take every 50th sample)
            traj_size: Number of consecutive windows per item (e.g., traj_size=2)
            traj_stride: Sample offset between windows (e.g., 800 = 4 seconds)
        """
```

### Loading & Preprocessing

```python
# Inside __init__:
for fname in file_list:
    mat = spio.loadmat(os.path.join(mat_dir, fname))
    data = mat['Data']
    
    # Extract signals and labels
    eeg_O1 = data.eeg_O1.astype(np.float32)           # (483200,)
    eeg_O2 = data.eeg_O2.astype(np.float32)           # (483200,)
    labels = data.labels_O1.astype(np.int64)          # (480000,) ← shorter due to annotation window
                                                       #   The first and last 1600 samples of the raw
                                                       #   signal are unannotated edge samples; labels
                                                       #   cover only the inner 480,000 annotated samples.
    
    # Pad so we can extract windows without edge effects
    # If we want window centered at sample 1600, we need samples [0, 3200]
    # Without padding, accessing sample 0-1600 from start would be invalid
    eeg_O1_padded = np.pad(eeg_O1, (w_len, w_len), mode='constant', constant_values=0)
    eeg_O2_padded = np.pad(eeg_O2, (w_len, w_len), mode='constant', constant_values=0)
    # eeg_O1_padded shape: (483200 + 2*1600,) = (486400,)
    
    # Pre-normalize: divide by 100, clip to [-1, 1]
    # Avoids per-sample normalization in __getitem__ (faster)
    eeg_O1_padded = np.clip(eeg_O1_padded / 100.0, -1.0, 1.0)
    eeg_O2_padded = np.clip(eeg_O2_padded / 100.0, -1.0, 1.0)
    
    # Store and accumulate counts for index mapping
    self.signals_O1.append(eeg_O1_padded)
    self.signals_O2.append(eeg_O2_padded)
    self.labels.append(labels)
    
    n_raw = len(labels)
    n_strided = (n_raw + stride - 1) // stride  # ceil division: e.g., 480000/50 = 9600
    self.subject_n_samples.append(n_strided)
    self.sample_offsets.append(total_samples)
    total_samples += n_strided

# Result: 76 subjects × 9600 strided samples = 729,600 total samples
```

### Windowing Strategy

**Why windows?** EEG context matters. A 1-sample prediction is unstable; 16 seconds of context helps.

**Why 16 seconds?** 
- Sleep stage patterns (sleep spindles, K-complexes) occur at that timescale
- Typical sleep architecture cycles: ~90 minutes = cycles of N1→N2→N3→REM
- 16s is a sweet spot: enough context without capturing unrelated cycles

```
Raw signal at 200 Hz:
[0, 1, 2, ..., 483199] samples

Padding:
[-1600, ... -1] [0, 1, ..., 483199] [483200, ..., 484799]
^                ^                  ^
pad_left         original           pad_right

Windowing (center at sample 1600):
start = center - w_len       = 1600 - 1600 = 0
end   = center + w_len       = 1600 + 1600 = 3200
window = signal[0:3200]     ← 3200 samples = 16 seconds @ 200 Hz
```

### Trajectory Sampling

**Concept**: Instead of just one 16s window, grab multiple overlapping windows **4 seconds apart**.

```
Timeline (time axis, samples):
0    800  1600 2400 3200  ...
|    |    |    |    |
[====window1====]          ← 3200 samples (16s)
     [====window2====]     ← 3200 samples, starts 800 samples later
          [====window3====]← If traj_size=3

traj_size=2:  2 windows, offset by traj_stride=800 samples (4s @ 200 Hz)
traj_size=3:  3 windows, each offset by 800 samples

Why? MWT events (e.g., microsleep) last median 4.6s. By sampling every 4s,
most events appear in multiple trajectory slots at different phases (onset, peak, offset).
Allows RayBNN to observe event dynamics across time steps.
```

### __getitem__ Implementation

```python
def __getitem__(self, idx):
    # idx ranges from 0 to total_samples
    # Example: total_samples = 729,600 (76 subjects × 9,600 strided samples each)
    
    # Step 1: Find which subject this index belongs to
    subj_idx = np.searchsorted(self.sample_offsets[1:], idx, side='right')
    # If idx=45000:
    #   sample_offsets might be [0, 9600, 19200, 28800, ...]
    #   searchsorted finds that 45000 is in subject 5 (after offset 28800)
    
    # Step 2: Map strided index to raw sample index
    strided_idx = idx - self.sample_offsets[subj_idx]  # e.g., 45000 - 28800 = 16200
    sample_idx = int(strided_idx * self.stride)         # e.g., 16200 * 50 = 810,000
    sample_idx = min(sample_idx, len(self.labels[subj_idx]) - 1)  # Clamp to valid range
    
    # Step 3: Extract trajectory windows
    windows = []
    labels_out = []
    for t in range(self.traj_size):
        # Offset by t * traj_stride for trajectory slots
        s_idx = min(sample_idx + t * self.traj_stride, max_idx)
        
        # Extract 16-second window centered at s_idx
        center = s_idx + self.w_len  # Adjust for padding
        start  = center - self.w_len
        end    = center + self.w_len  # = start + 3200
        
        # Get both channels
        window_O1 = self.signals_O1[subj_idx][start:end]  # (3200,)
        window_O2 = self.signals_O2[subj_idx][start:end]  # (3200,)
        
        # Stack channels and add width dimension for Conv2D
        window = np.stack([window_O1, window_O2], axis=0)  # (2, 3200)
        window = window[:, :, np.newaxis]                  # (2, 3200, 1)
        windows.append(window)
        labels_out.append(self.labels[subj_idx][s_idx])
    
    # Step 4: Format output
    if self.traj_size == 1:
        # Backward compatibility: single window → squeeze trajectory dim
        return torch.from_numpy(windows[0]).float(), \
               torch.tensor(labels_out[0], dtype=torch.long)
        # Shapes: (2, 3200, 1), scalar label
    else:
        # Multiple windows → stack into trajectory
        windows_arr = np.stack(windows, axis=0)   # (traj_size, 2, 3200, 1)
        labels_arr  = np.array(labels_out, dtype=np.int64)  # (traj_size,)
        return torch.from_numpy(windows_arr).float(), torch.from_numpy(labels_arr)
        # Shapes: (traj_size, 2, 3200, 1), (traj_size,)
```

> **Note**: RayBNN internally requires `traj_steps = traj_size + proc_num - 1` time slots
> (including warm-up slots). The reshaping from `traj_size` windows to `traj_steps` slots
> happens inside `AutoGradEndtoEnd`. See [Model Components](04_MODEL_COMPONENTS.md) for details.

**Index mapping visualization**:
```
Global index (DataLoader step):
0      1      2      ...    9599    9600             729599
│      │      │             │       │                │
Subject 0     Subject 0     Subject 0...           Subject 75
(strided_idx 0)             (strided_idx 9599)     (strided_idx 9599)

Raw sample mapping (stride=50):
idx=0    → strided_idx=0   → sample_idx=0
idx=1    → strided_idx=1   → sample_idx=50
idx=100  → strided_idx=100 → sample_idx=5000
idx=9600 → strided_idx=0   → sample_idx=0 (subject 1)
```

---

## 4-Fold Cross-Validation Splits

### Why Subject-Independent?

```python
def make_4fold_splits(mat_dir, seed=42):
    """
    Create subject-disjoint folds to prevent data leakage.
    
    WRONG approach:
        Shuffle all 729,600 samples → split 80/20 train/test
        ISSUE: Same subject appears in train AND test (overfitting)
    
    RIGHT approach (implemented here):
        Shuffle 76 subjects → split into 4 groups
        Fold k: test=group_k (~19 subjects), train=other 3 (~57 subjects)
    """
```

### Fold Creation

```python
all_files = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat')])
# all_files = ['subj_000.mat', 'subj_001.mat', ..., 'subj_075.mat'] (76 total)

rng = np.random.RandomState(seed=42)  # Fixed seed for reproducibility
rng.shuffle(all_files)  # Shuffle in-place

n = 76
n_folds = 4
# Distribute subjects evenly: 76 / 4 = 19 per fold

fold_sizes = [19, 19, 19, 19]  # All equal in this case
folds = [
    all_files[0:19],    # Fold 0 subjects: subj_X, subj_Y, ... (19 files)
    all_files[19:38],   # Fold 1 subjects
    all_files[38:57],   # Fold 2 subjects
    all_files[57:76]    # Fold 3 subjects
]

# For each fold k: test_files = fold_k, train_files = all others
splits = [
    {'train': folds[1]+folds[2]+folds[3], 'test': folds[0]},  # Fold 1
    {'train': folds[0]+folds[2]+folds[3], 'test': folds[1]},  # Fold 2
    {'train': folds[0]+folds[1]+folds[3], 'test': folds[2]},  # Fold 3
    {'train': folds[0]+folds[1]+folds[2], 'test': folds[3]},  # Fold 4
]
```

### Results
```
Fold 1: train=57 subjects (~460k samples), test=19 subjects (~153k samples)
Fold 2: train=57 subjects (~460k samples), test=19 subjects (~153k samples)
Fold 3: train=57 subjects (~460k samples), test=19 subjects (~153k samples)
Fold 4: train=57 subjects (~460k samples), test=19 subjects (~153k samples)
```

---

## DataLoader Batching

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1000,           # RayBNN fixed batch size
    shuffle=True,              # Randomize per epoch
    num_workers=4,             # 4 parallel processes for data loading
    pin_memory=True,           # Pre-allocate GPU memory for faster transfer
    drop_last=True,            # Drop incomplete final batch (ensures all batches are exactly 1000)
    persistent_workers=True    # Reuse workers across epochs (faster)
)
```

**Batch sizes context**:
- stride=50: 9,600 strided samples/subject × 57 subjects = 547,200 training samples
- 547,200 / 1,000 = 547 batches per epoch
- Each batch: (1000, 2, 3200, 1) or (1000, traj_size, 2, 3200, 1) if traj_size > 1

**Memory**:
- Batch tensor: 1,000 × 2 × 3,200 × 1 × 4 bytes (float32) ≈ 25.6 MB
- DataLoader pre-loads ~4 batches: ~100 MB resident in RAM

---

## Class Imbalance Problem

### The Issue

```python
# Compute label distribution from full training set
all_labels = np.concatenate(train_dataset.labels)
label_counts = np.bincount(all_labels, minlength=4)
# Output example:
# [431000, 12000, 8000, 4000]
# Wake^    N1^    N2^   Microsleep^
# Ratios: Wake:N1 = 36:1, Wake:Microsleep = 108:1 !!!
```

**Problem**: MSE/CE loss treats all classes equally
- Model learns: "Just predict Wake always" → 95% accuracy
- But Microsleep predictions = always zero (useless)
- Kappa = 0 (predicts 1 class, label 4 classes)

### Solution: Weighted Loss

```python
# Class weights inversely proportional to frequency
class_weights = 1.0 / np.sqrt(label_counts)
# sqrt instead of direct inverse: less extreme, more stable
# Example:
# Wake:      1 / sqrt(431000) ≈ 0.0048
# N1:        1 / sqrt(12000)  ≈ 0.0091
# N2:        1 / sqrt(8000)   ≈ 0.0112
# Microsleep: 1 / sqrt(4000)  ≈ 0.0158

# Normalize so mean = 1 (and sum = 4)
class_weights = class_weights / class_weights.sum() * 4
# After normalization: [0.55, 1.05, 1.30, 1.83]
# → Microsleep gets 3.3× more loss than Wake!

# Use in loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

# Effect: Minority class misclassifications → much larger loss gradients
# Model motivated to learn minority patterns despite lower frequency
```

---

## Data Flow Summary

```
Training iteration:
  for batch_x, batch_y in train_loader:
      batch_x.shape = (1000, 2, 3200, 1) | (1000, traj_size, 2, 3200, 1) if traj_size > 1
      batch_y.shape = (1000,)             | (1000, traj_size)             if traj_size > 1
      
      # Example walk-through:
      # - Loader calls dataset[random_indices] (1000 times in parallel)
      # - Each call: fetch subject, extract window, format as torch tensor
      # - Stack into batch
      # - Move to GPU: batch_x.to(device)
      # 
      # Model forward:
      batch_x → CNN (12 layers) → features (1000, 256) → RayBNN → logits (1000, 4)
      
      # Loss:
      loss = CrossEntropyLoss(weight=class_weights)(logits, batch_y)  # Weighted!
      
      # Backward:
      loss.backward()  # Uses AutoGradEndtoEnd
```

---

## Next Steps

- **See how data flows through CNN**: [Model Components](04_MODEL_COMPONENTS.md)
- **Training mechanics**: [Training Flow](05_TRAINING_FLOW.md)
- **Run end-to-end**: [Running the Script](07_RUNNING_THE_SCRIPT.md)

---

[← Back to README](README.md) | [Previous: Architecture](02_ARCHITECTURE.md) | [Next: Model Components →](04_MODEL_COMPONENTS.md)

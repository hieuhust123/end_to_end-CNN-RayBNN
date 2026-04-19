# 7. Running the Script: Step-by-Step Guide

[← Back to README](README.md)

## Prerequisites

### 1. Required Dependencies

```bash
# Python 3.8+
python --version
# Output: Python 3.x.x ✓

# Check CUDA (optional but recommended)
nvidia-smi
# If error: CUDA not installed (will fall back to CPU, much slower)
```

### 2. Python Packages

```bash
# Install from requirements.txt (if available)
pip install -r ../requirements.txt

# Or install manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib
pip install raybnn_python  # The Rust-based BNN library
```

### 3. Data Setup

```
Expected directory structure:
RayBNN_Python/
├── Python_Code/
│   └── mwt_test_backward_cnn+raybnn_testing.py  ← The script
├── mwt_eeg/
│   ├── subj_000.mat
│   ├── subj_001.mat
│   └── ... (76 total .mat files)
└── DOCUMENTATION/
    └── (you are here)
```

**Verify data is present**:
```bash
ls ../mwt_eeg/*.mat | wc -l
# Output: 76  ← Good! All 76 subjects present

# Check file sizes (~6-7 MB each)
du -h ../mwt_eeg/subj_000.mat
# Output: 6.5M  ← Good!
```

---

## Running the Script

### Mode 1: Full Run (Recommended for First Time)

```bash
cd /home/hbui/Downloads/RayBNN_Python/Python_Code

# Run the script
python mwt_test_backward_cnn+raybnn_testing.py

# Expected: Runs 4-fold CV, prints lots of diagnostics, takes 60-120 min
```

**What you'll see**:

```
[Output Epoch 1 - lots of initialization]
MWT EEG data directory: ../mwt_eeg
MWT dataset: 76 subjects, 480,000 raw samples, stride=50 → 960,000 effective samples

======================================================================
  4-FOLD CROSS-VALIDATION — Fold 1 / 4
======================================================================
[Fold 1] Using device: cuda  ← GPU detected ✓
[Fold 1] Loading training dataset (57 subjects)...
[Fold 1] Loading test dataset (19 subjects)...
MWTDataset: 57 subjects, 320,000 raw samples, stride=50 → 640,000 effective samples
MWTDataset: 19 subjects, 160,000 raw samples, stride=50 → 320,000 effective samples

======================================================================
 TRAINING: 50 epochs, batch_size=1000
 alpha0=0.0005, lr=0.0003, weight_decay=1e-4, scheduler=CosineAnnealing(eta_min=0.000003)
======================================================================

Epoch 1/50 - loss: 1.2345 - acc: 0.5612 - test_loss: 1.1234 - test_acc: 0.6123 - kappa: 0.4567 - lr: 0.000300
Epoch 2/50 - loss: 1.0456 - acc: 0.6234 - test_loss: 0.9876 - test_acc: 0.6789 - kappa: 0.5234 - lr: 0.000299
...
Epoch 50/50 - loss: 0.3456 - acc: 0.8901 - test_loss: 0.4567 - test_acc: 0.8654 - kappa: 0.7654 - lr: 0.000003

[Fold 1] Done — acc=0.8654  kappa=0.7654  time=20.1 min

======================================================================
  4-FOLD CROSS-VALIDATION — Fold 2 / 4
======================================================================
... (repeat 3 more times)

======================================================================
  4-FOLD CROSS-VALIDATION — FINAL SUMMARY
======================================================================
Fold      Accuracy    Kappa       F1       Time (min)
────────────────────────────────────────────────────
  1       0.8234      0.7123      0.7956      20.5
  2       0.8156      0.6987      0.7821      19.2
  3       0.8289      0.7234      0.8045      21.1
  4       0.8145      0.7001      0.7889      20.2
────────────────────────────────────────────────────
  Mean    0.8206      0.7086      0.7928      20.3
  Std     0.0063      0.0106      0.0092

Concatenated kappa: 0.7089
Total wall-clock time: 81.5 min
======================================================================
Done without errors!
```

**Output files created**:
```bash
ls -la *.png
# mwt_cnn_raybnn_fold1_loss.png
# mwt_cnn_raybnn_fold2_loss.png
# mwt_cnn_raybnn_fold3_loss.png
# mwt_cnn_raybnn_fold4_loss.png
```

---

### Mode 2: Reduced Run (Testing)

If you want to test the pipeline without waiting 2 hours:

```python
# Edit line in mwt_test_backward_cnn+raybnn_testing.py:

# BEFORE (line ~1650):
max_epoch = 50

# AFTER:
max_epoch = 3  # Just 3 epochs for quick test

# Save and run:
python mwt_test_backward_cnn+raybnn_testing.py
# Expected: Completes in ~10-15 minutes, still produces all outputs
```

---

### Mode 3: Single Fold Debug

To debug a single fold without running full 4-fold CV:

```python
# Edit __main__ section at bottom of script:

if __name__ == '__main__':
    mat_dir = '../mwt_eeg'
    fold_splits = make_4fold_splits(mat_dir, seed=42)
    
    # Only run fold 0
    fold_idx = 0
    split = fold_splits[fold_idx]
    
    end_to_end_model, train_dataset, test_dataset, alpha0, batch_size, device = \
        main(split['train'], split['test'], mat_dir, fold_idx=fold_idx)
    
    trained_model, train_history = train_ete_model(
        end_to_end_model, train_dataset, test_dataset,
        alpha0, batch_size=batch_size, max_epoch=50, mode="both",
        fold_idx=fold_idx
    )
    
    results = evaluate_model(trained_model, test_dataset, batch_size=batch_size)
    plot_losses(train_history, save_path="fold0_loss.png")
    
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"Kappa: {results['kappa_overall']:.4f}")
    print("Done!")

# Save and run:
python mwt_test_backward_cnn+raybnn_testing.py
# Expected: Runs 1 fold only, completes in ~20 minutes
```

---

## Monitoring Training Progress

### Real-Time Checks

While training is running, in another terminal:

```bash
# Check which fold is running
grep "FOLD" mwt_test_backward_cnn+raybnn_testing.py | head -1
# Or watch tail of output:
tail -f any_output.txt  # if redirected to file

# Check GPU usage (if running on GPU)
watch -n 1 nvidia-smi
# Should show:
#   a python process using ~12 GB VRAM
#   GPU utilization ~90-100%

# Check system memory usage
top -p $(pgrep -f "python.*mwt_test")
# Should show RAM usage ~peak of 8-12 GB
```

### Expected Timing

```
Per-fold breakdown (on modern GPU like RTX 3090):
  Data loading:        ~5 minutes (first fold)
  Training 50 epochs:  ~15-20 minutes
  Evaluation:          ~2-3 minutes
  Total per fold:      ~20-25 minutes

Full 4-fold CV:        ~90-120 minutes (1.5-2 hours)

If taking much longer:
  • CPU-only mode (no GPU): 8-12 hours (very slow!)
  • Older GPU: 2-4 hours (acceptable)
  • Laptop/limited RAM: May crash (see troubleshooting)
```

### Saving Output to File

```bash
# Redirect all output to file (can take ~500 MB)
python mwt_test_backward_cnn+raybnn_testing.py > training_output.txt 2>&1 &

# Monitor output in real-time
tail -f training_output.txt

# Or save + print
python mwt_test_backward_cnn+raybnn_testing.py | tee training_output.txt
```

---

## Understanding Output

### Per-Epoch Output

```
Epoch 25/50 - loss: 0.5234 - acc: 0.8123 - test_loss: 0.6789 - test_acc: 0.7654 - kappa: 0.6123 - lr: 0.000153 [best=0.6400@ep23, stale=1/12]
 │            │         │               │                    │                    │            │                   └─ Early stopping info
 │            │         │               │                    │                    │            └─ Current learning rate
 │            │         │               │                    │                    └─ Cohen's kappa (main metric)
 │            │         │               │                    └─ Test accuracy
 │            │         │               └─ Test loss (unweighted CE)
 │            │         └─ Training accuracy
 │            └─ Training loss (weighted CE)
 └─ Epoch number

Interpretation:
  • loss: 0.5234 → Decreasing from previous epoch ✓ (learning)
  • acc: 0.8123 → Good training accuracy, but typically < test_acc
  • test_loss: 0.6789 → Higher than train loss (normal, test is harder)
  • kappa: 0.6123 → Pretty good kappa (0.6+), closer to "excellent"
  • stale=1/12 → 1 eval since last improvement, patience=12 → plenty of time
```

### When Things Go Wrong

**Loss explosion** (loss jumps to NaN):
```
Epoch 3/50 - loss: 0.3201 - acc: 0.8521
Epoch 4/50 - loss: NaN - acc: NaN
⚠️ LOSS EXPLOSION DETECTED: nan
Stopping training to prevent further damage

Likely cause:
  • Learning rate too high (alpha0 or CNN lr)
  • Unstable gradient (check AREA 2 gradients are normal)
```

**Loss stuck/very high**:
```
Epoch 1/50: loss: 2.5023
Epoch 2/50: loss: 2.4891
Epoch 3/50: loss: 2.4756
⚠️ Model not learning!

Likely cause:
  • Learning rate too low
  • Data not loading correctly
  • Model architecture broken
```

**Memory error (OOM)**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MB

Likely cause:
  • Batch size too large (use smaller batch_size)
  • Gradient checkpointing disabled (enable it)
  • Data leaking into graph (check detach/no_grad usage)

Fix:
  • Reduce batch_size from 1000 to 500
  • Check if VRAM is actually available (nvidia-smi)
```

---

## Extracting Results Programmatically

### Post-Processing

```python
import numpy as np
import torch

# After script completes, results are in fold_results list
# To access from outside the script:

# Option 1: Save results to JSON (add to end of script)
import json

results_summary = {
    "fold_accuracies": [r['accuracy'] for r in fold_results],
    "fold_kappas": [r['kappa_overall'] for r in fold_results],
    "mean_accuracy": np.mean([r['accuracy'] for r in fold_results]),
    "mean_kappa": np.mean([r['kappa_overall'] for r in fold_results]),
    "concatenated_kappa": concat_kappa,
}

with open('results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Then access from another script:
with open('results.json', 'r') as f:
    results = json.load(f)
    print(results)
```

### Extracting Training Curves

```python
# Save training history for plotting
import matplotlib.pyplot as plt

fold_idx = 0
history = fold_histories[fold_idx]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Train loss
axes[0, 0].plot(history['epoch_losses'])
axes[0, 0].set_title('Train Loss')

# Plot 2: Test loss
axes[0, 1].plot(history['test_losses'])
axes[0, 1].set_title('Test Loss')

# Plot 3: Accuracy
axes[1, 0].plot(history['epoch_accs'], label='train')
axes[1, 0].plot(history['test_accs'], label='test')
axes[1, 0].set_title('Accuracy')
axes[1, 0].legend()

# Plot 4: Test Kappa
axes[1, 1].plot(history['test_kappas'])
axes[1, 1].set_title('Test Kappa')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=100)
plt.show()
```

---

## Common Issues & Fixes

### Issue 1: "Module 'raybnn_python' not found"

```
Error: ModuleNotFoundError: No module named 'raybnn_python'

Fix:
  1. Install RayBNN Python bindings:
     pip install raybnn_python
  
  2. If that fails, build from source:
     cd ../raybnn/  # or wherever RayBNN source is
     pip install .
  
  3. Verify installation:
     python -c "import raybnn_python; print('OK')"
```

### Issue 2: MATLAB File Loading Error

```
Error: scipy.io.loadmat() fails to load .mat files

Fix:
  1. Verify .mat files exist:
     ls -la ../mwt_eeg/ | head -5
  
  2. Check file format:
     file ../mwt_eeg/subj_000.mat
     # Should say: "Matlab 5.0 MAT-file" or similar
  
  3. Try loading manually in Python:
     import scipy.io as spio
     mat = spio.loadmat('../mwt_eeg/subj_000.mat')
     print(mat.keys())  # Should show: ['Data', '__header__', ...]
  
  4. If MATLAB version is very old or corrupted:
     Re-export from MATLAB with: save(..., '-v7') or '-v7.3'
```

### Issue 3: CUDA Out of Memory (OOM)

```
RuntimeError: CUDA out of memory. Tried to allocate 25.00 GiB.

Fix:
  1. Reduce batch_size. Edit line in main():
     batch_size = 1000  → change to 500 or 256
     (Note: RayBNN may require changing its initialization too)
  
  2. Enable gradient checkpointing (already enabled for blocks 1-7)
  
  3. Clear GPU cache before running:
     import torch
     torch.cuda.empty_cache()
  
  4. If all else fails, run on CPU (very slow):
     device = torch.device('cpu')
```

### Issue 4: Training Freezes / Hangs

```
Script appears to hang (CPU/GPU usage drops to 0)

Fix:
  1. Check if stuck in data loading:
     • DataLoader num_workers too high (try num_workers=0)
     • Add verbosity: print() statements in MWTDataset.__getitem__
  
  2. Check if stuck in RayBNN forward/backward:
     • Add print() before/after raybnn_python calls
     • Reduce max_epoch for debugging
  
  3. Kill and restart:
     pkill -f "python.*mwt_test"
```

### Issue 5: Very Different Results Between Runs

```
Fold 1: kappa=0.72
Fold 2: kappa=0.65  ← Much lower
Fold 3: kappa=0.71
Fold 4: kappa=0.68

Why?
  • Randomness in fold splits: different subjects → different difficulty
  • Randomness in weight initialization: different random seed → different convergence
  • Randomness in batch order: shuffling within epoch differences

Fix:
  • Set seed for reproducibility:
    import random; import numpy as np; import torch
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  • Add at top of script, before main()
```

---

## Next Runs: Customization

### Modify Hyperparameters

```python
# Edit in main() function:

# Learning rates
alpha0 = 0.0005  # RayBNN LR, try: 0.0001, 0.0010
# CNN LR is set in train_ete_model() as 0.0003

# Batch size (may affect memory requirements)
batch_size = 1000  # Try: 500, 256 (smaller = less memory, slower)

# Trajectory sampling
traj_size = 2  # Try: 1 (single window), 3 (three overlapping windows)
traj_stride = 800  # Try: 400 (more overlap), 1600 (less overlap)

# Training epochs
max_epoch = 50  # Try: 30 (shorter), 100 (longer)

# Data sampling
stride=50  # Try: 10 (more samples, slower), 100 (fewer, faster)
```

### Modify Model Architecture

```python
# Edit CNN_EEG class init():

# More/fewer Conv filters
self.conv1 = nn.Conv2d(2, 64, kernel_size=(3,1), padding=(1,0))  # 2->32 -> 2->64
# Higher = more capacity, more memory, risk of overfitting

# Add dropout layers
self.drop_conv1 = nn.Dropout(p=0.3)  # Before each block

# Change activation (ReLU -> GELU)
x = F.gelu(self.bn1(self.conv1(x)))  # Instead of F.relu

# Change normalization (BatchNorm -> LayerNorm)
self.ln1 = nn.LayerNorm((32, 3200))  # Instead of BatchNorm2d
```

---

## Next Steps

- **Understand results**: [Evaluation](06_EVALUATION.md)
- **Understand code**: [Architecture](02_ARCHITECTURE.md)
- **Debug training**: [Training Flow](05_TRAINING_FLOW.md)

---

[← Back to README](README.md) | [Previous: Evaluation](06_EVALUATION.md)

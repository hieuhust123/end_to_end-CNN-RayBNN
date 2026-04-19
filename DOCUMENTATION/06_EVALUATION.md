# 6. Evaluation: Metrics & Diagnostics

[← Back to README](README.md)

## Evaluation Metrics

### Primary Metric: Cohen's Kappa

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
# Range: [-1, 1]
#   1.0 = perfect agreement
#   0.5 = moderate agreement
#   0.0 = random chance
#  -1.0 = perfect disagreement
```

**Why kappa?** Accuracy alone is misleading with class imbalance.

```
Example with imbalanced data:
  95% Wake, 5% other classes

Bad model:
  "Always predict Wake"
  Accuracy = 95% ✗ (sounds great, but useless)
  Kappa = 0.0 ✓ (reveals it's chance performance)

Good model:
  "Predict Wake 95%, other classes 5%"
  Accuracy = 75% (lower, but honest)
  Kappa = 0.60 ✓ (shows real generalization)
```

**Interpretation**:
```
Kappa ≥ 0.81: Excellent
0.61 - 0.80:  Good
0.41 - 0.60:  Moderate
0.21 - 0.40:  Fair
< 0.20:       Slight/Poor
```

### Secondary Metrics

```python
# Accuracy: % of correct predictions
accuracy = (predictions == labels).mean()
# Range: [0, 1], useful as sanity check

# Precision: Of positive predictions, how many are correct?
#   precision[i] = TP[i] / (TP[i] + FP[i])
# Focus: False positives
precision = precision_score(labels, predictions, average='macro')

# Recall: Of true positives, how many did we catch?
#   recall[i] = TP[i] / (TP[i] + FN[i])
# Focus: False negatives
recall = recall_score(labels, predictions, average='macro')

# F1: Harmonic mean of precision & recall
#   F1 = 2 * (precision * recall) / (precision + recall)
f1 = f1_score(labels, predictions, average='macro')

# Loss: Cross-entropy loss (lower is better)
#   Reduces as model becomes more confident (regardless of accuracy)
loss = criterion(logits, labels)
```

### Per-Class Metrics

```python
# Kappa per class (one-vs-rest)
def kappa_metric(y_true, y_pred, n_cl=4):
    """Computes Cohen kappa treating each class as binary."""
    y = np.array(y_true)
    y_ = np.array(y_pred)
    res = []
    for c in range(n_cl):
        # For class c: y==c is "positive", else "negative"
        res.append(cohen_kappa_score(y == c, y_ == c))
    return np.array(res)

# Example output:
# [0.85, 0.70, 0.62, 0.45]
#   Wake: 0.85 (excellent)
#   N1:   0.70 (good)
#   N2:   0.62 (moderate)
#   MS:   0.45 (fair, hardest to detect)
```

### Confusion Matrix Interpretation

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
# cm[i,j] = number of samples with true label i, predicted label j
# Diagonal = correct predictions
# Off-diagonal = errors

# Example:
#         Pred: Wake  N1    N2    MS
# True:
#   Wake   9000   50    30    20    (mostly correct)
#   N1      100   800   80    20    (some confusion with N2)
#   N2       50   100   600   150   (confused with N1 & MS)
#   MS       20    40   100    50    (very hard to predict)

# Reading:
#   Diagonal (9000, 800, 600, 50): % correct per class
#   Row 3 (N2): 50+100+150=250 errors out of 800 samples = 69% accuracy
#   Col 4 (MS pred): mostly off-diagonal = model overpredicts MS elsewhere
```

---

## 4-Fold Cross-Validation Results

### Per-Fold Reporting

```python
print(f"\n{'='*70}")
print(f" 4-FOLD CROSS-VALIDATION — FINAL SUMMARY")
print(f"{'='*70}")
print(f"{'Fold':<8} {'Accuracy':>10} {'Kappa':>10} {'F1':>10} {'Time (min)':>12}")
print(f"{'-'*52}")
for r in fold_results:
    k = r['fold_idx']
    print(f"  {k:<6} {r['accuracy']:>10.4f} {r['kappa_overall']:>10.4f} "
          f"{r['f1_score']:>10.4f} {r['fold_time_sec']/60:>12.1f}")
print(f"{'-'*52}")
print(f"  {'Mean':<6} {np.mean(accs):>10.4f} {np.mean(kappas):>10.4f} "
      f"{np.mean(f1s):>10.4f} {np.mean(times)/60:>12.1f}")
print(f"  {'Std':<6} {np.std(accs):>10.4f} {np.std(kappas):>10.4f} "
      f"{np.std(f1s):>10.4f}")

# Example output:
# Fold      Accuracy    Kappa       F1       Time (min)
# ───────────────────────────────────────────────────────
#   1       0.8234      0.7123      0.7956      18.5
#   2       0.8156      0.6987      0.7821      19.2
#   3       0.8289      0.7234      0.8045      20.1
#   4       0.8145      0.7001      0.7889      19.8
# ───────────────────────────────────────────────────────
#   Mean    0.8206      0.7086      0.7928      19.4
#   Std     0.0063      0.0106      0.0092
```

### Concatenated Kappa (Xuan Chen Style)

```python
# Pool all predictions across folds
all_preds = np.concatenate([fold_results[i]['_raw_preds'] for i in range(4)])
all_labels = np.concatenate([fold_results[i]['_raw_labels'] for i in range(4)])

# Single kappa on pooled data
concat_kappa = cohen_kappa_score(all_labels, all_preds)
# Example: 0.7089
# (Typically slightly different from mean kappa due to class distribution shifts)
```

### Reporting Conventions

```
Academic paper / conference:
  "We achieved 82.06% accuracy with Cohen's kappa = 0.7086 ± 0.0106
   across 4-fold subject-independent cross-validation."

Clinical setting:
  "Model agreement with expert labels (kappa) indicates good-to-excellent
   performance, with room for improvement on minority classes (N2, MS)."
```

---

## Training Diagnostics (AREA 1-4)

These are verbose output blocks printed during training to validate the pipeline.

### AREA 1: CNN Parameter Updates

**Purpose**: Verify CNN weights are changing (learning is happening).

```
AREA 1 DIAGNOSTIC: Checking CNN Parameter Updates
======================================================================

=== Initial CNN Parameters ===
conv1.weight:
  Shape: [32, 2, 3, 1]
  Mean: 0.021543
  Std: 0.084231
  Min/Max: [-0.245123, 0.198456]

bn1.weight:
  Shape: [32]
  Mean: 1.000000
  Std: 0.000000
  Min/Max: [1.000000, 1.000000]

... (all 12 layers) ...

Memory before training: 2543.2 MB
```

**What to check**:
- ✓ Initial weights are small (near zero) — good initialization
- ✓ BatchNorm weights start at 1.0 — expected
- ✓ Shapes match architecture (see [Architecture](02_ARCHITECTURE.md))

---

### AREA 2: CNN Gradient Analysis

**Purpose**: Verify backprop produced non-zero gradients.

```
======================================================================
AREA 2 DIAGNOSTIC: CNN Parameters Gradients AFTER backprop (Epoch 0, Batch 0)
======================================================================

=== CNN Gradients ===

conv1.weight:
  Gradient mean: 0.00032456
  Gradient max: 0.01543
  Gradient min: 0.00001
  Healthy gradient magnitude

conv1.bias:
  Gradient mean: 0.00089234
  Gradient max: 0.02134
  Gradient min: 0.00005
  Healthy gradient magnitude

bn1.weight:
  Gradient mean: 0.00156234
  ... (rest similar)

=== CNN Parameter Delta (Epoch 1, after optimizer.step) ===
  conv1.weight: mean_delta=0.00000312, max_delta=0.00004543
  bn1.weight: mean_delta=0.00000089, max_delta=0.00001234
  ...
```

**Interpretation**:
- ✓ Gradients are non-zero → backprop is working
- ✓ Gradients are small (1e-4 to 1e-3) → reasonable for Adam
- ⚠️ Gradients are zero → backprop didn't flow (check for detach, requires_grad, etc.)
- ⚠️ Gradients are huge (> 1.0) → likely loss explosion (check for NaN handling)

---

### AREA 3: CNN Output Features

**Purpose**: Verify CNN produces diverse, useful representations.

```
=== CNN Output (Features) BEFORE entering RayBNN ===
  Shape: (1000, 256)
  Mean: 0.156234
  Std: 0.823456
  Min: -2.345678
  Max: 3.456789
  Variance across samples (per feature): 0.123456
  Variance across features (per sample): 0.987654
  Features vary across samples ✓
```

**Interpretation**:
- ✓ Features have non-zero mean & std → not all zeros
- ✓ Per-sample variance is high → different inputs produce different features (discriminative)
- ⚠️ Per-sample variance < 1e-6 → all samples identical features (model hasn't learned anything)
- ⚠️ Min/Max extreme (-10, +10) → instability, possible gradient explosion upstream

---

### AREA 4: RayBNN Output Analysis

**Purpose**: Verify RayBNN produces reasonable, confident predictions.

```
======================================================================
AREA 4 DIAGNOSTIC: RayBNN Yhat (CNN->RayBNN->Softmax input) (Epoch 1, Batch 0)
======================================================================

Output Statistics:
  Shape: (1000, 4)
  Range: [-1.2345, 3.4567]
  Mean: 0.234567
  Std: 0.789012
  
Entropy: 1.2143/1.3863 (random=1.3863)
GOOD: Moderate entropy - learning in progress
    ↑ max_entropy(4 classes) = ln(4) ≈ 1.386
    ↑ Entropy measures prediction confidence
    ↑ 1.214 < 1.386 means model has learned preferences
```

**Entropy interpretation**:
```
Entropy ≈ 1.386 (max): Model outputs ≈ uniform [0.25, 0.25, 0.25, 0.25]
                       WARNING: Predictions are near-random!

Entropy ≈ 0.7 (med):  Model has some confidence
                     GOOD: Learning in progress

Entropy < 0.3 (low):   Model is very confident
                      EXCELLENT (if confident AND accurate)
                      PROBLEM (if confident and wrong = overfit)
```

---

## Loss & Accuracy Plots

### Generated Plots

For each fold, the script saves:
```
mwt_cnn_raybnn_fold1_loss.png    ← PNG with 2 subplots
    Subplot 1: Train vs Test Loss across epochs
    Subplot 2: Test Cohen Kappa across epochs
```

### Interpreting Loss Curves

```
HEALTHY training (expected):
              Test loss
              ╱  ╲  ╲
            ╱      ╲  ╲
          ╱          ╲  ╲
Train    ╱ ╲           ╲  ╲
loss:  ╱    ╲           ╲   ╲ (converged)
                
      └──────────────────────── epoch
      
Characteristics:
  • Both losses decrease, especially early
  • Test loss > train loss (expected, test set harder)
  • After ~epoch 30, changes flatten (convergence)
  • Smooth curves (no spikes)

OVERFITTING (detected by early stopping):
Train loss ↘ (keeps improving)
Test loss  ↘↗ (improves, then worsens) ← Stop here!

UNDERFITTING:
Loss ╱ ╱ (both stuck high, no improvement)
     ← Not enough model capacity or learning is too slow

LOSS EXPLOSION:
Loss ↑↑↑↑↑ very fast, may show NaN
     ← Usually: learning rate too high, gradient explosion
```

---

## Kappa Interpretation for Sleep Stages

### Realistic Expectations

```
Perfect manual annotation agreement: kappa ≈ 0.80
(Sleep experts themselves disagree sometimes)

Good ML model:
  Wake:       0.85 (easier, longer epochs)
  N1:         0.70 (transitory, hard to distinguish from N2)
  N2:         0.65 (majority class, good discrimination possible)
  Microsleep: 0.45 (rare, hard to detect without eye tracking)
  Overall:    0.70

Why different per class?
  • Wake: Long durations, clear EEG patterns → easy
  • MS: Short (2-30s), rare → hard to sample and learn
  • N1: Transitional, similar to Wake/N2 → confusing
```

---

## Throughput & Runtime Analysis

### Memory Profiling

```python
import psutil

process = psutil.Process()
mem_before = process.memory_info().rss / 1024**2  # MB

# ... training ...

mem_after = process.memory_info().rss / 1024**2
delta = mem_after - mem_before

print(f"Memory: before={mem_before:.1f} MB, after={mem_after:.1f} MB, delta={delta:.1f} MB")
# Expected: delta should be < 100 MB (mostly garbage collection)
# If delta > 500 MB: memory leak (graph not released, gradients accumulating)
```

### Throughput Analysis

```
Training statistics (per fold):
  Samples per epoch: 460,000  (from 57 subjects × 9,600 strided samples each)
  Batches per epoch:  460
  Epochs completed:   50
  Total training time: 1200 seconds (20 min)
  
  Throughput = (460,000 samples × 50 epochs) / 1200 sec = 19,167 samples/sec
  Per-batch time = 1200 sec / (460 batches × 50 epochs) = 0.104 sec/batch
  Batch size = 1000 samples
  Per-sample time = 0.104 sec / 1000 = 0.000104 sec ≈ 0.1 ms/sample ✓
  
Expected on modern GPU:
  • 12 GB VRAM GPU: 50-100 ms/batch (1000 samples) ✓
  • CPU only: 500-2000 ms/batch (very slow) ✗
```

---

## Next Steps

- **How to run**: [Running the Script](07_RUNNING_THE_SCRIPT.md)
- **Debugging issues**: [Training Flow](05_TRAINING_FLOW.md) (diagnostics section)

---

[← Back to README](README.md) | [Previous: Training Flow](05_TRAINING_FLOW.md) | [Next: Running the Script →](07_RUNNING_THE_SCRIPT.md)

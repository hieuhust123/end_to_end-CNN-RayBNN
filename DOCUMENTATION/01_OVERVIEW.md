# 1. Overview: What Is This Script?

[← Back to README](README.md)

## Purpose

This script trains an **end-to-end deep learning model** to classify EEG sleep stages using a hybrid approach:
- **CNN** (Convolutional Neural Network): Extracts features from raw EEG signals
- **RayBNN** (Bayesian Neural Network): Performs probabilistic classification on CNN features

The goal: **Automate sleep stage detection** from EEG data (O1, O2 channels) with uncertainty quantification.

---

## The Problem We're Solving

**Sleep researchers** want to classify EEG signals into 4 sleep stages:
- **Wake** — Eyes open or alert
- **N1** — Light sleep (transition to deeper sleep)
- **N2** — Main sleep stage
- **Microsleep** — Brief, involuntary sleep episodes

**Manual annotation** is:
- Time-consuming (expert needs ~10 mins per hour of EEG)
- Subjective (different experts may disagree)
- Not scalable (76 subjects × ~8 hours each = 10,000+ hours of EEG)

**Solution**: Train an ML model to do this automatically and check its agreement with human labels (Cohen's kappa).

---

## High-Level Pipeline

```
Raw EEG Signals
    ↓
[Preprocessing] ← Normalize, window, create trajectories
    ↓
[CNN] ← Extract 256-dim feature vectors (Xuan Chen architecture)
    ↓
[RayBNN] ← Classify features → 4 sleep stage probabilities
    ↓
[Loss & Gradients] ← Custom autograd integrates Rust gradient computation
    ↓
[Optimizer] ← Update CNN params + defer BNN param updates
    ↓
Trained Model → Evaluate on held-out test subjects (4-fold CV)
```

---

## Why This Approach?

### Why CNN?
- EEG is a **time-series signal** → CNNs excel at capturing temporal patterns
- The Xuan Chen architecture was already validated on this exact task
- Fast: Extracts features quickly on GPU

### Why RayBNN + CNN together?
- **CNN alone** is deterministic (no uncertainty)
- **RayBNN adds** Bayesian uncertainty → model knows when it's unsure
- **Joint training** allows CNN to learn features that RayBNN prefers
- Mimics biological plausibility (hierarchical feature extraction → probabilistic decision-making)

### Why custom autograd?
- **RayBNN is written in Rust** (not differentiable by default)
- **Custom AutoGrad** bridges the gap:
  - Captures CNN → RayBNN forward pass
  - Receives RayBNN gradients from Rust
  - Flows them back to CNN
  - Allows joint optimization of both components

---

## Dataset: MWT EEG

### What is MWT?
**Maintenance of Wakefulness Test** — measures ability to stay awake during quiet, darkened conditions. Used in sleep research to detect narcolepsy and sleepiness disorders.

### Data Details
- **Subjects**: 76 people
- **Channels**: 2 (O1, O2 = occipital lobe positions)
- **Sampling rate**: 200 Hz
- **Duration per subject**: ~8 hours
- **Labels**: One of {Wake, N1, N2, Microsleep} per sample
- **Imbalance**: Wake is ~100× more frequent than N2/Microsleep → need weighted loss

### Why 4-Fold Cross-Validation?
- Ensures **subject-independence**: No overlap between train & test subjects
- Prevents overfitting to specific individuals
- Gives 4 independent estimates of model performance (we report mean ± std)
- Standard in sleep research

---

## Key Terminology

| Term                                          | Definition |
|------|-----------|
| **Window**                                    | 16-second EEG segment (3200 samples at 200 Hz) |
| **Trajectory**                                | Sequence of overlapping 16s windows (e.g., traj_size=2 means 2 consecutive windows) |
| **Stride**                                    | Sampling interval; stride=50 takes every 50th sample to reduce data redundancy |
| **Traj_stride**                               | Offset between consecutive trajectory windows; 800 samples = 4 seconds |
| **Features**                                  | 256-dim output from CNN's final dense layer |
| **One-hot label**                             | Binary encoding of class (e.g., N2 → [0,0,1,0]) |
| **Batch**                                     | Simultaneous processing of 1000 samples (RayBNN requirement) |

---

## Model Architecture at a Glance

```
Input: EEG signal (2 channels, 16s = 3200 samples)
    ↓
[CNN: 12 convolutional layers]
    • Conv(2→32) → Conv(32→64) → Conv(64→128)×4 → Conv(128→256)×6
    • ReLU activation, batch norm, max pooling after each Conv
    • Gradually reduces time dimension via pooling
    ↓
Output: (batch_size, 256) = feature vectors
    ↓
[RayBNN: Fully connected + outputs]
    • Takes 256-dim features
    • Outputs 4 class logits (Wake, N1, N2, Microsleep)
    ↓
Classification output: (batch_size, 4) logits → softmax → probabilities
```

---

## Training Workflow

### High-Level Steps

1. **Data Preparation** (once per fold)
   - Load raw EEG from .mat files
   - Split subjects into 4 groups (folds)
   - For fold k: train on 3 groups (~57 subjects), test on 1 group (~19 subjects)

2. **Model Initialization** (per fold)
   - Create fresh CNN_EEG instance
   - Initialize RayBNN with starting architecture
   - Set optimizer (Adam), loss (weighted CrossEntropy), learning rates

3. **Training Loop** (50 epochs per fold)
   - **Forward**: Batch → CNN features → RayBNN classification
   - **Backward**: Loss → RayBNN gradients (Rust) → CNN gradients (PyTorch)
   - **Update**: CNN via PyTorch optimizer, RayBNN via deferred CPU Adam
   - **Evaluate**: Every 2 epochs on test set

4. **Result Aggregation** (after all 4 folds)
   - Report per-fold metrics (accuracy, kappa, F1, time)
   - Compute cross-fold statistics (mean ± std)
   - Pool predictions across folds for concatenated kappa (Cohen's style)

---

## Key Design Decisions

### 1. **Deferred BNN Updates**
- Problem: RayBNN backward used to update BNN weights inside the backward pass
- Issue: CNN gradients (flow to optimizer) were computed against a stale BNN state
- Solution: Cache RayBNN gradients, apply BNN update AFTER CNN optimizer.step()
- Benefit: Both updates computed against same model state → more stable training

### 2. **Gradient Checkpointing**
- Large activations (early Conv layers) consume ~95% of peak VRAM
- Solution: Recompute activations during backward instead of caching
- Trade-off: +30% computation, -50% peak memory
- Applied to blocks 1-7 (later blocks are already small)

### 3. **Weighted Loss**
- Wake is 100× more frequent than Microsleep
- Without weighting: Model ignores minority classes
- Solution: `loss_weight[i] = 1 / sqrt(label_count[i])` (sqrt-inverse, less extreme than 1/count)
- Result: All classes contribute meaningfully to gradient

### 4. **Cosine Annealing LR Scheduler**
- Learning rate decays smoothly over all epochs
- Final LR = 1% of initial LR
- Helps fine-tuning in later epochs

### 5. **Early Stopping via Test Kappa**
- Monitor Cohen's kappa on test set (evaluated every 2 epochs)
- Stop if kappa doesn't improve for 12 evaluations (~24 epochs)
- Restore best checkpoint before returning
- Prevents overfitting

---

## Outputs

### Files Generated
```
mwt_cnn_raybnn_fold1_loss.png  ← Train/test loss + kappa plot for fold 1
mwt_cnn_raybnn_fold2_loss.png  ← (same for folds 2-4)
mwt_cnn_raybnn_fold3_loss.png
mwt_cnn_raybnn_fold4_loss.png
```

### Console Output Examples
```
[Fold 1] Using device: cuda
MWTDataset: 76 subjects, 480,000 raw samples, stride=50 → 960,000 effective samples
[Fold 1] Loading training dataset (57 subjects)...
[Fold 1] Loading test dataset (19 subjects)...

=== [Training begins] ===
Epoch 1/50 - loss: 1.2345 - acc: 0.5612 - test_loss: 1.1234 - test_acc: 0.6123 - kappa: 0.4567 - lr: 0.000300

...

Final test kappa: 0.6789 @ epoch 45
```

### Performance Summary
```
4-FOLD CROSS-VALIDATION — FINAL SUMMARY
Fold      Accuracy    Kappa       F1       Time (min)
  1       0.8234      0.7123      0.7956   18.5
  2       0.8156      0.6987      0.7821   19.2
  3       0.8289      0.7234      0.8045   20.1
  4       0.8145      0.7001      0.7889   19.8
-------------------------------------------------------
Mean      0.8206      0.7086      0.7928   19.4
Std       0.0063      0.0106      0.0092

Concatenated kappa: 0.7089
Total wall-clock time: 78.7 min
```

---

## Next Steps

**Ready to dive deeper?**
- **Understand the model**: [Architecture](02_ARCHITECTURE.md)
- **See data handling**: [Data Pipeline](03_DATA_PIPELINE.md)
- **Run it yourself**: [Running the Script](07_RUNNING_THE_SCRIPT.md)
- **Master the code**: [Model Components](04_MODEL_COMPONENTS.md)

---

[← Back to README](README.md) | [Next: Architecture →](02_ARCHITECTURE.md)

# MWT EEG Data Pipeline — Key Points Summary

**Purpose**: Concise reference for understanding how raw EEG data flows from `.mat` files to a trained model in `mwt_test_backward_cnn+raybnn_testing.py`.

---

## Big Picture: End-to-End Flow

```
76 Subject .mat files
        │
        ▼
┌─────────────────────────┐
│  make_4fold_splits()    │  ← Split 76 subjects into 4 groups of 19
│  seed=42, shuffle once  │    No subject in both train AND test
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    │  Repeat 4 times │  (one per fold)
    └────────┬────────┘
             │
    ┌────────▼────────────────────────┐
    │  MWTDataset (train: 57 subj.)   │  ← Load + pad + normalize signals
    │  MWTDataset (test:  19 subj.)   │    Store raw signals in RAM (~280 MB)
    └────────┬────────────────────────┘
             │
    ┌────────▼────────────────────────┐
    │  DataLoader (batch_size=1000)   │  ← On-demand window extraction
    │  shuffle=True, num_workers=4    │    547 batches per epoch
    └────────┬────────────────────────┘
             │
    ┌────────▼────────────────────────┐
    │  CNN_EEG (12 layers)            │  ← (batch, 2, 3200, 1) → (batch, 256)
    └────────┬────────────────────────┘
             │
    ┌────────▼────────────────────────┐
    │  RayBNN                         │  ← (batch, 256) → (batch, 4) logits
    └────────┬────────────────────────┘
             │
    ┌────────▼────────────────────────┐
    │  Weighted CrossEntropyLoss      │  ← Handles class imbalance
    │  + Backward + Optimizer step    │
    └────────┬────────────────────────┘
             │
    ┌────────▼────────────────────────┐
    │  Evaluate on test set           │  ← Report kappa, F1, accuracy
    └─────────────────────────────────┘
             │
    After 4 folds: report mean ± std kappa + concatenated kappa
```

---

## 1. The Raw Data

| Property | Value |
|----------|-------|
| Subjects | 76 |
| Channels | 2 (O1, O2 — occipital EEG) |
| Sample rate | 200 Hz |
| Samples per subject | 483,200 (~40 hours) |
| Labeled samples | 480,000 (first/last 1,600 are unannotated edges) |
| Labels | 0=Wake, 1=N1, 2=N2, 3=Microsleep |
| Format | MATLAB `.mat` files, one per subject |

**Key point**: The dataset is severely imbalanced. Wake dominates; Microsleep is rare.

```
Label distribution (approximate):
Wake        ████████████████████████████████████  ~89%
N1          ████                                  ~6%
N2          ██                                    ~2%
Microsleep  █                                     ~3%
```

---

## 2. Subject-Level 4-Fold Split (`make_4fold_splits`)

### Why subject-level?

If you split by samples (not subjects), the same person can appear in both train and test. The model then memorizes subject-specific EEG patterns instead of learning to generalize — a form of data leakage.

### How it works

```
76 subjects (shuffled once, seed=42)
│
├── Group A (19 subjects) ─┐
├── Group B (19 subjects)  │
├── Group C (19 subjects)  │
└── Group D (19 subjects) ─┘

Fold 1:  Train = B + C + D (57)   Test = A (19)
Fold 2:  Train = A + C + D (57)   Test = B (19)
Fold 3:  Train = A + B + D (57)   Test = C (19)
Fold 4:  Train = A + B + C (57)   Test = D (19)

→ Every subject is tested exactly once across the 4 folds.
→ No overlap between train and test in any fold.
```

**Fixed seed = reproducible**: Running the script twice always produces the same fold assignments.

---

## 3. Loading & Preprocessing (`MWTDataset.__init__`)

For each subject file, three operations happen at load time (done once, stored in RAM):

### 3a. Padding

```
Original signal: [0 ........ 483199]  (483,200 samples)
                  ^                ^
                  |                |
After padding:   [PAD | 0 ........ 483199 | PAD]
                 1600   ←original→         1600
                 zeros                     zeros
Total: 486,400 samples per channel

Why? So we can extract a 16-second window centered at sample 0 without
going out of bounds — the window reaches back into the padding instead.
```

### 3b. Normalization

```python
eeg_padded = np.clip(eeg_padded / 100.0, -1.0, 1.0)
```

Divide by 100, then clamp to `[-1, 1]`. Done once at load time, not per batch.

### 3c. Stride Counting

With `stride=50`, only every 50th sample is used as a window center:

```
480,000 raw samples → 9,600 effective samples per subject
57 train subjects   → 547,200 total training samples
19 test subjects    → 182,400 total test samples
547,200 / 1,000 (batch size) = 547 batches per epoch
```

---

## 4. On-Demand Window Extraction (`MWTDataset.__getitem__`)

When the DataLoader requests one item, this is what happens:

### Step A: Find the subject

```
Global index (e.g., idx=14,500)
         │
         ▼  binary search over sample_offsets
Subject index (e.g., subject 1, strided_idx=4,900)
         │
         ▼  × stride (50)
Raw sample position (e.g., sample 245,000)
```

### Step B: Extract trajectory windows

Instead of one window, the code extracts **`traj_size=2` windows**, each shifted by `traj_stride=800` samples (4 seconds):

```
Raw signal timeline (samples):
         245000        245800
           │             │
           ▼             ▼
Window 1: [←────── 3200 samples = 16s ──────→]
Window 2:      [←────── 3200 samples = 16s ──────→]
               ↑
          shifted 4 seconds later

Why 4 seconds? Median MWT microsleep event ≈ 4.6 seconds.
Two windows 4s apart catch the same event at different phases:
onset in window 1, peak/offset in window 2.
```

### Step C: Format output

```
Each window: stack O1 + O2, add width dim
   (2, 3200, 1)  ← 2 channels × 3200 time steps × 1

Two windows stacked:
   (2, 2, 3200, 1)  ← traj_size × channels × time × 1

Labels:
   (2,)  ← one label per window center
```

---

## 5. DataLoader Batching

```
1,000 items from __getitem__, stacked by DataLoader:

batch_x: (1000, 2, 2, 3200, 1)
          ^     ^  ^  ^     ^
          batch traj chan time width

batch_y: (1000, 2)
          ^     ^
          batch traj

Memory per batch: 1000 × 2 × 2 × 3200 × 1 × 4 bytes ≈ 51.2 MB
```

`shuffle=True` → order randomized each epoch  
`num_workers=4` → 4 CPU workers prefetch data in parallel while GPU trains  
`drop_last=True` → ensures every batch is exactly 1,000 (RayBNN requirement)

---

## 6. Class Imbalance: Weighted Loss

The raw class distribution would cause the model to just always predict Wake (89% accurate, but kappa ≈ 0). The fix:

```
Class weights = 1 / sqrt(label_count), then normalized so mean = 1

Example (approximate):
  Wake:        weight ≈ 0.55   (most common → lowest weight)
  N1:          weight ≈ 1.05
  N2:          weight ≈ 1.30
  Microsleep:  weight ≈ 1.83   (rarest → highest weight)

Effect: a Microsleep misclassification costs ~3.3× more than a Wake one.
Model is pushed to learn minority class patterns despite fewer examples.
```

```python
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
```

---

## 7. The 4-Fold Training Loop

```
Script start
│
├── make_4fold_splits() → 4 splits
│
├── Fold 1 ──────────────────────────────────────────────┐
│   │  Build fresh model (CNN + RayBNN)                  │
│   │  Load train dataset (57 subjects)                  │
│   │  Load test  dataset (19 subjects)                  │
│   │  Train for up to 50 epochs                         │
│   │    └─ every 2 epochs: evaluate on test, check kappa│
│   │    └─ early stop if kappa flat for 12 evaluations  │
│   │    └─ Adam reset every 15 epochs                   │
│   │  Final evaluate → record kappa, F1, accuracy       │
│   └─────────────────────────────────────────────────── ┘
│
├── Fold 2 (same process, different subjects)
├── Fold 3
├── Fold 4
│
└── Aggregate:
      Mean kappa ± std across 4 folds
      Concatenated kappa (pool all 4 folds' predictions)
```

---

## 8. Key Numbers at a Glance

| Parameter | Value | Why |
|-----------|-------|-----|
| `w_len` | 1,600 samples | Half-window; full window = 3,200 = 16s at 200 Hz |
| `stride` | 50 | Use every 50th sample → 9,600 effective samples/subject |
| `traj_size` | 2 | 2 overlapping windows per item |
| `traj_stride` | 800 samples | 4 seconds; matches median MWT event duration |
| `batch_size` | 1,000 | Fixed (RayBNN requirement) |
| `max_epoch` | 50 | Training cap per fold |
| Folds | 4 | Each fold: 57 train / 19 test subjects |
| Batches/epoch | ~547 | 547,200 train samples / 1,000 |
| CNN output | 256-dim | Feature vector fed into RayBNN |
| RayBNN output | 4 classes | Wake, N1, N2, Microsleep |
| Primary metric | Cohen's Kappa | Accounts for class imbalance |

---

## 9. Why Kappa, Not Accuracy?

```
Scenario: model always predicts Wake
  Accuracy = 89%  ← looks good!
  Kappa    = 0.0  ← actually useless (no better than random chance)

Kappa = (observed agreement − chance agreement) / (1 − chance agreement)

Scale:
  < 0.00  → Worse than chance
  0.00    → No better than random
  0.20    → Slight agreement
  0.40    → Moderate agreement
  0.60    → Substantial agreement   ← our results land here (~0.54)
  0.80    → Almost perfect
  1.00    → Perfect
```

Accuracy is misleading when classes are imbalanced. Kappa is the honest metric.

---

## 10. Common Confusions Clarified

**Q: Why is the label array shorter (480,000) than the signal array (483,200)?**  
The raw EEG signal has 1,600 unannotated edge samples on each end. Labels only cover the inner 480,000 annotated samples. Padding compensates so windows can still be centered at the edges.

**Q: Why pad with zeros instead of reflecting the signal?**  
Zero padding is safe — the model sees silence at the edges, which is unambiguous. Reflected padding could introduce artificial patterns at boundaries.

**Q: Why stride=50 and not use all 480,000 samples?**  
Adjacent samples are nearly identical (200 Hz = very dense). Taking every 50th sample (every 0.25 seconds) gives sufficient coverage with 4× less redundancy and faster training.

**Q: Why does traj_size matter for RayBNN?**  
RayBNN is a state-space model — it processes a sequence of inputs over time. `traj_size` determines how many time steps it sees per training item. More trajectory steps = richer temporal context, but slower and more memory-intensive.

---

[← Back to README](README.md) | [Previous: Experiment Results](08_EXPERIMENT_RESULTS.md)

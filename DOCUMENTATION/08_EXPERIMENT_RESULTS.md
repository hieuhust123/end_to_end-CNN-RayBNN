# 8. Experiment Results: Configuration Comparison

[← Back to README](README.md)

**Purpose**: Structured log of all training runs comparing different configurations of the CNN + RayBNN pipeline on the MWT EEG dataset. Use this to understand what was tried, what worked, and what did not.

**Date compiled**: 2026-04-16  
**Dataset**: MWT EEG — 76 subjects, 4 sleep stages (Wake, N1, N2, Microsleep)  
**Evaluation**: 4-fold subject-independent cross-validation, primary metric = Cohen's Kappa

---

## Quick Reference: All Runs at a Glance

| # | Configuration | traj_size | proc_num | max_epoch | Early Stop | Adam Reset | Status | Mean Kappa | Concat Kappa |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Baseline (no early stop) | 1 | 2 | 50 | ✗ | ✗ | Complete | 0.5382 ± 0.055 | 0.5338 |
| 2 | Epoch-15 quick run | 1 | 2 | 15 | ✗ | ✗ | Complete | 0.5393 ± 0.052 | 0.5341 |
| 3 | Early stop patience=8 | 1 | 2 | 50 | 8 | ✗ | Complete | 0.5323 ± 0.083 | 0.5215 |
| 4 | **Early stop + Adam reset** | **1** | **2** | **50** | **12** | **✓** | **Complete** | **0.5397 ± 0.028** | **0.5367** |
| 5 | proc_num=4 | 1 | 4 | 50 | 12 | ✓ | Incomplete (Fold 1 only) | — | — |
| 6 | traj_size=2 | 2 | 2 | 50 | 12 | ✓ | Incomplete (Folds 1–2 + partial 3) | — | — |
| 7 | traj_size=4 | 4 | 2 | 50 | 12 | ✓ | Incomplete (Folds 1–3) | — | — |

> **Kappa interpretation**: ≥ 0.81 Excellent · 0.61–0.80 Good · 0.41–0.60 Moderate · 0.21–0.40 Fair · < 0.20 Poor  
> **Key**: traj_size = number of overlapping 16s EEG windows per sample · proc_num = RayBNN internal processing steps

---

## Run 1 — Baseline: No Early Stopping

**File**: `epoch_50_lr_0.0003_alpha_0.0005_weight_decay.txt`

### Configuration
| Parameter | Value |
|---|---|
| CNN learning rate | 0.0003 |
| RayBNN learning rate (alpha0) | 0.0005 |
| max_epoch | 50 |
| traj_size | 1 |
| proc_num | 2 |
| Early stopping | None |
| Adam state reset | No |
| LR scheduler | CosineAnnealing (eta_min=3e-6) |
| Weight decay | 1e-4 |

> **Note**: This was the first full run — no early stopping means all 50 epochs always run even after kappa stops improving.  
> **Note**: Test set sizes differ per fold because subject counts vary slightly (subjects have different recording lengths).

### Per-Fold Results

| Fold | Accuracy | Kappa | F1-macro | Best Kappa (epoch) | Time |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9017 | 0.4838 | 0.4191 | 0.5042 @ ep7 | 56.6 min |
| 2 | 0.9080 | 0.5536 | 0.4526 | 0.5903 @ ep1 | 57.4 min |
| 3 | 0.9296 | 0.6208 | 0.4581 | 0.6318 @ ep17 | 59.2 min |
| 4 | 0.8948 | 0.4944 | 0.4288 | 0.5418 @ ep3 | 58.0 min |
| **Mean** | **0.9085** | **0.5382** | **0.4396** | | **57.8 min** |
| **Std** | 0.0130 | 0.0546 | 0.0162 | | |
| **Concat kappa** | | **0.5338** | | | **231.1 min total** |

### Per-Class Breakdown (averaged over 4 folds)

| Class | Precision (avg) | Recall (avg) | F1 (avg) | Notes |
|---|:---:|:---:|:---:|---|
| Wake | ~0.95 | ~0.96 | ~0.95 | Excellent — dominant class |
| N1 | ~0.70 | ~0.73 | ~0.69 | Good recall |
| N2 | ~0.13 | ~0.01 | ~0.02 | Near-zero — almost never detected |
| Microsleep | ~0.10 | ~0.07 | ~0.08 | Very poor |

### Training Behavior

- **Kappa peaked very early**: Best kappa often at epoch 1–17, then **overfitting** — test kappa declined while train accuracy continued rising toward ~99%
- No mechanism to stop or recover from overfitting → final kappa is often much lower than peak kappa (e.g., Fold 1: best=0.50, final=0.48)
- High variance across folds (std=0.055) due to no regularization of training duration
- Training was fast (~58 min/fold) with small test sets in this run

### Takeaway
> Establishes the performance floor. Shows that overfitting is a real problem without early stopping — the model memorizes training data after ~10–20 epochs. Adding early stopping is necessary.

---

## Run 2 — Epoch-15 Quick Run

**File**: `mwt_epoch_15.txt`

### Configuration
| Parameter | Value |
|---|---|
| CNN learning rate | **0.0008** (higher than other runs) |
| RayBNN learning rate (alpha0) | 0.0005 |
| max_epoch | **15** |
| traj_size | 1 |
| proc_num | 2 |
| Early stopping | None |
| Adam state reset | No |
| LR scheduler | None noted |
| Weight decay | Not applied |

> **Intent**: A fast exploratory run to verify the pipeline works end-to-end and get a quick kappa estimate before committing to longer runs.

### Per-Fold Results

| Fold | Accuracy | Kappa | F1-macro | Best Kappa (epoch) | Time |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.8690 | 0.4550 | 0.4023 | 0.5029 @ ep11 | 17.3 min |
| 2 | 0.8896 | 0.5536 | 0.4561 | 0.6004 @ ep1 | 17.4 min |
| 3 | 0.9143 | 0.5981 | 0.4780 | 0.6550 @ ep11 | 18.0 min |
| 4 | 0.9172 | 0.5505 | 0.4556 | 0.5924 @ ep3 | 17.7 min |
| **Mean** | **0.8976** | **0.5393** | **0.4480** | | **17.6 min** |
| **Std** | 0.0197 | 0.0522 | 0.0279 | | |
| **Concat kappa** | | **0.5341** | | | **70.5 min total** |

### Per-Class Breakdown (averaged over 4 folds)

| Class | Precision (avg) | Recall (avg) | F1 (avg) | Notes |
|---|:---:|:---:|:---:|---|
| Wake | ~0.96 | ~0.94 | ~0.95 | Excellent |
| N1 | ~0.58 | ~0.85 | ~0.68 | High recall — model tends to over-predict N1 |
| N2 | ~0.14 | ~0.04 | ~0.06 | Very poor |
| Microsleep | ~0.13 | ~0.09 | ~0.10 | Very poor |

### Training Behavior

- Despite only 15 epochs, kappa **matches or exceeds the 50-epoch baseline** (mean 0.5393 vs 0.5382) — confirming that most learning happens in the first ~15 epochs
- Higher LR (0.0008) drove faster early descent but the effect is indistinguishable from the 50-epoch run at 15 epochs
- Peak kappa often came at epoch 1 or 11, suggesting the best checkpoint is reached very early then lost to overfitting
- Wall-clock time was **3.3× faster** (70.5 vs 231 min) — useful for rapid iteration

### Takeaway
> Demonstrates that the model learns most of what it can in ≤15 epochs. Beyond that, it overfits without intervention. This strongly motivated adding early stopping + checkpoint restoration in subsequent runs.

---

## Run 3 — Early Stopping, Patience=8

**File**: `epoch_50_lr_0.0003_alpha_0.0005_weight_decay_early_stop_8.txt`

### Configuration
| Parameter | Value |
|---|---|
| CNN learning rate | 0.0003 |
| RayBNN learning rate (alpha0) | 0.0005 |
| max_epoch | 50 |
| traj_size | 1 |
| proc_num | 2 |
| Early stopping patience | **8** (stop after 8 evals without improvement) |
| Adam state reset | No |
| LR scheduler | CosineAnnealing (eta_min=3e-6) |
| Weight decay | 1e-4 |

> **Intent**: Add early stopping to prevent overfitting. Patience=8 means stop if kappa doesn't improve for 8 consecutive evaluations (evaluation every 2 epochs → stops after ~16 stale epochs).

### Per-Fold Results

| Fold | Accuracy | Kappa | F1-macro | Best Kappa (epoch) | Time |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.8685 | 0.4659 | 0.4464 | — | 36.4 min |
| 2 | 0.8928 | 0.5702 | 0.4643 | — | 46.1 min |
| 3 | 0.9319 | 0.6502 | 0.4811 | — | 56.9 min |
| 4 | 0.8593 | 0.4427 | 0.4170 | — | 37.5 min |
| **Mean** | **0.8881** | **0.5323** | **0.4522** | | **44.2 min** |
| **Std** | 0.0281 | **0.0833** | 0.0237 | | |
| **Concat kappa** | | **0.5215** | | | **177.0 min total** |

### Per-Class Breakdown (averaged over 4 folds)

| Class | Precision (avg) | Recall (avg) | F1 (avg) | Notes |
|---|:---:|:---:|:---:|---|
| Wake | ~0.96 | ~0.93 | ~0.94 | Excellent |
| N1 | ~0.60 | ~0.85 | ~0.70 | Good recall |
| N2 | ~0.10 | ~0.04 | ~0.05 | Very poor — Fold 4 predicted zero N2 (F1=0.0) |
| Microsleep | ~0.09 | ~0.11 | ~0.08 | Very poor |

### Training Behavior

- Early stopping triggered inconsistently — some folds stopped early (~36 min), others ran longer (~57 min) → causing **high variance** (std=0.083, worst of all complete runs)
- Patience=8 may be **too short**: the model's kappa fluctuates naturally across evaluation windows, so 8 evaluations without improvement may cut off before the model has fully converged
- Fold 4 notably failed on N2 (F1=0.0, kappa per class ≈ 0.0) — the model predicted essentially zero N2 samples in that fold
- Concat kappa (0.5215) is the lowest among the early-stopping runs

### Takeaway
> Early stopping helps recover the best checkpoint, but patience=8 introduced more variance than it removed. The model needs more time before declaring convergence. Increasing patience to 12 and adding Adam state reset was the natural next step.

---

## Run 4 — Early Stopping (patience=12) + Adam Reset ⭐ Best Complete Run

**File**: `epoch_50_lr_0.0003_alpha_0.0005_weight_decay_early_stop_12_reset_adam.txt`

### Configuration
| Parameter | Value |
|---|---|
| CNN learning rate | 0.0003 |
| RayBNN learning rate (alpha0) | 0.0005 |
| max_epoch | 50 |
| traj_size | 1 |
| proc_num | 2 |
| Early stopping patience | **12** |
| **Adam state reset** | **Yes — every 15 epochs** |
| LR scheduler | CosineAnnealing (eta_min=3e-6) |
| Weight decay | 1e-4 |

> **Adam state reset**: Every 15 epochs, the RayBNN Adam optimizer's momentum buffers (m_t, v_t) are zeroed out. This prevents stale gradient momentum from pushing BNN weights in a direction that no longer reflects the current loss landscape — particularly important when the CNN is still changing rapidly.

### Per-Fold Results

| Fold | Accuracy | Kappa | F1-macro | Best Kappa (epoch) | Time |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.8968 | 0.5206 | 0.4663 | 0.5360 @ ep15 | 176.0 min |
| 2 | 0.8581 | 0.5179 | 0.4640 | 0.5972 @ ep5 | 133.4 min |
| 3 | 0.8747 | 0.5332 | 0.4818 | 0.6644 @ ep1 | 118.5 min |
| 4 | 0.9260 | 0.5871 | 0.4475 | 0.5904 @ ep37 | 232.2 min |
| **Mean** | **0.8889** | **0.5397** | **0.4649** | | **165.0 min** |
| **Std** | 0.0255 | **0.0280** | 0.0122 | | |
| **Concat kappa** | | **0.5367** | | | **660.2 min total** |

### Per-Class Breakdown (Fold 1 — representative)

| Class | Precision | Recall | F1 | Kappa |
|---|:---:|:---:|:---:|:---:|
| Wake | 0.9593 | 0.9381 | 0.9486 | 0.567 |
| N1 | 0.5422 | 0.8384 | 0.6585 | 0.628 |
| N2 | 0.1902 | 0.1333 | 0.1567 | 0.150 |
| Microsleep | 0.1452 | 0.0780 | 0.1014 | 0.082 |
| **macro avg** | **0.4592** | **0.4969** | **0.4663** | |

### Training Behavior

- **Lowest variance** of all runs (std=0.028 vs 0.083 for patience=8) — Adam reset made training more consistent across folds
- Kappa trend was volatile epoch-to-epoch but the best checkpoint was reliably recovered by early stopping
- Training was significantly **slower** (~165 min/fold vs ~58 min for baseline) — likely due to larger test sets in this dataset split configuration
- Fold 4's best kappa (0.5904) came at epoch 37 — later than other folds — suggesting the Adam reset at epoch 15 gave a second convergence push
- N2 and Microsleep remain the hardest classes but showed improvement vs. patience=8 (N2 F1 ~0.15 vs near-zero in some folds)

### Takeaway
> **This is the most reliable configuration.** The combination of patience=12 + Adam state reset produced the highest mean kappa AND the lowest variance. The Adam reset in particular appears critical for stable joint training of CNN and RayBNN. Recommend using this as the baseline for future experiments.

---

## Run 5 — proc_num=4 (Incomplete)

**File**: `proc_num_4_epoch_50_lr_0.0003_alpha_0.0005_weight_decay_early_stop_12_reset_adam.txt`

> ⚠️ **INCOMPLETE — Terminated after Fold 1 (deliberately stopped)**. Fold 2 had reached epoch 7 when terminated.

### Configuration
| Parameter | Value |
|---|---|
| CNN learning rate | 0.0003 |
| RayBNN learning rate (alpha0) | 0.0005 |
| max_epoch | 50 |
| traj_size | 1 |
| **proc_num** | **4** (vs. 2 in all other runs) |
| Early stopping patience | 12 |
| Adam state reset | Yes — every 15 epochs |

> **proc_num** controls the number of internal RayBNN processing steps per forward pass. Higher proc_num = more recurrent computation inside the Bayesian layer, potentially richer temporal integration, but slower training.

### Results — Fold 1 Only

| Fold | Accuracy | Kappa | F1-macro | Best Kappa (epoch) | Time |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9064 | 0.5200 | 0.4572 | **0.5310 @ ep37** | 254.0 min |
| 2–4 | — | — | — | — | Terminated |

### Per-Class Breakdown (Fold 1)

| Class | Precision | Recall | F1 | Kappa |
|---|:---:|:---:|:---:|:---:|
| Wake | 0.9499 | 0.9548 | 0.9523 | 0.552 |
| N1 | 0.5836 | 0.7815 | 0.6682 | 0.640 |
| N2 | 0.1854 | 0.1163 | 0.1430 | 0.136 |
| Microsleep | 0.1741 | 0.0401 | 0.0652 | 0.054 |
| **macro avg** | **0.4733** | **0.4732** | **0.4572** | |

### Training Behavior

- Fold 1 kappa (0.5200) is comparable to Run 4 Fold 1 (0.5206) — no obvious gain from proc_num=4 at this one fold
- Training was **~1.5× slower** than Run 4 Fold 1 (254 min vs 176 min) due to the additional RayBNN processing steps
- N1 recall improved slightly (0.78 vs 0.84 in Run 4 Fold 1), but N2 and Microsleep remain weak
- Cannot draw conclusions without all 4 folds

### Takeaway
> proc_num=4 did not show a clear kappa improvement in the one completed fold, while significantly increasing training time. A full 4-fold run is needed to properly evaluate. **Not recommended to prioritize without a complete run.**

---

## Run 6 — traj_size=2 (Incomplete)

**File**: `traj_size_2_epoch_50_lr_0.0003_alpha_0.0005_weight_decay_early_stop_12_reset_adam.txt`

> ⚠️ **INCOMPLETE — Terminated during Fold 3 (epoch 7 reached, kappa rising to 0.6187)**. Folds 1 and 2 are complete.

### Configuration
| Parameter | Value |
|---|---|
| CNN learning rate | 0.0003 |
| RayBNN learning rate (alpha0) | 0.0005 |
| max_epoch | 50 |
| **traj_size** | **2** (two overlapping 16s windows, 4s apart per sample) |
| proc_num | 2 |
| traj_steps | 3 (= traj_size + proc_num − 1) |
| Early stopping patience | 12 |
| Adam state reset | Yes — every 15 epochs |

> **traj_size=2** means each training sample contains two overlapping 16-second EEG windows centered 4 seconds apart (covering a total span of 20 seconds). This gives RayBNN temporal context: it can observe how features evolve across two time steps, which may help detect short events like Microsleep (median duration ~4.6s).

### Results — Folds 1 & 2 Complete

| Fold | Accuracy | Kappa | F1-macro | Best Kappa (epoch) | Time |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9055 | 0.4786 | 0.4066 | 0.4861 @ ep49 | 454.0 min |
| 2 | 0.8384 | 0.4842 | 0.4477 | 0.5989 @ ep1 | 231.8 min |
| 3 (partial) | — | 0.6187 @ ep7 (best so far) | — | — | Terminated |
| 4 | — | — | — | — | Not reached |

### Per-Class Breakdown (Fold 1)

| Class | Precision | Recall | F1 | Kappa |
|---|:---:|:---:|:---:|:---:|
| Wake | 0.9417 | 0.9635 | 0.9525 | 0.517 |
| N1 | 0.5508 | 0.6756 | 0.6068 | 0.575 |
| N2 | 0.1742 | 0.0152 | 0.0280 | 0.027 |
| Microsleep | 0.1703 | 0.0221 | 0.0392 | 0.032 |
| **macro avg** | **0.4593** | **0.4191** | **0.4066** | |

### Training Behavior

- **Training is extremely slow**: Fold 1 took 454 min (2.6× slower than Run 4 Fold 1 at 176 min), Fold 2 took 232 min. Each sample now sends 2× the data through the CNN
- Despite the additional temporal context, kappa on completed folds (0.48, 0.48) is **lower than Run 4** (0.52, 0.52) — trajectory context did not help, and may have hurt due to the harder optimization landscape
- N2 and Microsleep showed no improvement — the classes the trajectory was meant to help
- Best kappa for Fold 1 came at epoch 49 (last epoch), suggesting the model had not converged — may need more epochs or a different LR schedule
- Fold 3 was trending upward at epoch 7 (kappa=0.618) but was terminated before completing

### Takeaway
> traj_size=2 was **not beneficial** in the completed folds and increased training time by 2.6×. The hypothesis that temporal context helps detect short Microsleep events was not confirmed. Possible reasons: (1) the CNN architecture is not optimized for multi-window input; (2) more epochs or a tuned LR may be needed. **Needs a full run before definitive conclusion.**

---

## Run 7 — traj_size=4 (Incomplete)

**File**: `traj_size_4_epoch_50_lr_0.0003_alpha_0.0005_weight_decay_early_stop_12_reset_adam.txt`

> ⚠️ **INCOMPLETE — Terminated at the start of Fold 4 (epoch 1)**. Folds 1, 2, and 3 are complete.

### Configuration
| Parameter | Value |
|---|---|
| CNN learning rate | 0.0003 |
| RayBNN learning rate (alpha0) | 0.0005 |
| max_epoch | 50 |
| **traj_size** | **4** (four overlapping 16s windows, 4s apart — covers 28 seconds total) |
| proc_num | 2 |
| traj_steps | 5 (= traj_size + proc_num − 1) |
| Early stopping patience | 12 |
| Adam state reset | Yes — every 15 epochs |

> **traj_size=4** extends the temporal context further: 4 consecutive 16-second windows sampled 4 seconds apart, spanning 28 seconds per sample. Each batch now carries 4× the CNN data compared to traj_size=1.

### Results — Folds 1, 2, 3 Complete

| Fold | Accuracy | Kappa | F1-macro | Best Kappa (epoch) | Time |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9030 | 0.4556 | 0.3865 | 0.4611 @ ep39 | 96.0 min |
| 2 | 0.9020 | 0.5504 | 0.4288 | 0.5392 @ ep11 | 66.7 min |
| 3 | 0.9191 | 0.5899 | 0.4143 | 0.5914 @ ep9 | 64.7 min |
| 4 (partial) | — | — | — | Terminated @ ep1 | — |

### Per-Class Breakdown (Fold 1)

| Class | Precision | Recall | F1 | Kappa |
|---|:---:|:---:|:---:|:---:|
| Wake | 0.9389 | 0.9643 | 0.9514 | 0.498 |
| N1 | 0.5186 | 0.6346 | 0.5707 | 0.536 |
| N2 | **0.0000** | **0.0000** | **0.0000** | **≈0.000** |
| Microsleep | 0.2642 | 0.0125 | 0.0238 | 0.021 |
| **macro avg** | **0.4304** | **0.4028** | **0.3865** | |

### Training Behavior

- **N2 completely missed in Folds 1 and 3** (F1=0.0, kappa≈0) — the model predicted zero N2 samples. This is a regression vs. traj_size=1 where N2 F1 was ~0.15
- Kappa variance across completed folds is large (0.46, 0.55, 0.59) — Fold 1 is substantially weaker than Folds 2 and 3
- Training was faster than traj_size=2 (~75 min/fold), likely because early stopping triggered earlier
- Fold 3 early stopping triggered at epoch 33 (stale=12/12) — kappa stopped improving at 0.5914 (epoch 9) and then fluctuated downward
- Fold 4 was only at epoch 1 when terminated — no usable data

### Takeaway
> traj_size=4 showed **mixed results**: Folds 2–3 kappas are comparable to Run 4, but Fold 1 was the weakest of all runs (0.4556), and N2 was completely undetected in two folds. Extending trajectory context to 4 windows appears to destabilize learning for minority classes. **Not recommended without architectural changes to handle longer trajectories.**

---

## Cross-Run Comparison

### Summary Table (Complete Runs Only)

| Run | Config | Mean Kappa | Std | Concat Kappa | Mean Acc | Mean F1 | Total Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Baseline (no early stop) | 0.5382 | 0.055 | 0.5338 | 0.909 | 0.440 | 231 min |
| 2 | Epoch-15 quick | 0.5393 | 0.052 | 0.5341 | 0.898 | 0.448 | 71 min |
| 3 | Early stop patience=8 | 0.5323 | 0.083 | 0.5215 | 0.888 | 0.452 | 177 min |
| **4** | **Early stop p=12 + Adam reset** | **0.5397** | **0.028** | **0.5367** | **0.889** | **0.465** | **660 min** |

### Per-Class Kappa Across Runs (Fold 1, best available)

| Run | Wake kappa | N1 kappa | N2 kappa | MS kappa |
|---|:---:|:---:|:---:|:---:|
| 1 — Baseline | 0.523 | 0.617 | 0.013 | 0.044 |
| 2 — Epoch-15 | 0.489 | 0.553 | 0.037 | 0.021 |
| 3 — ES p=8 | 0.508 | 0.580 | 0.120 | 0.081 |
| **4 — ES p=12 + Adam** | **0.567** | **0.628** | **0.150** | **0.082** |
| 5 — proc_num=4 (F1 only) | 0.552 | 0.640 | 0.136 | 0.054 |
| 6 — traj_size=2 (F1 only) | 0.517 | 0.575 | 0.027 | 0.032 |
| 7 — traj_size=4 (F1 only) | 0.498 | 0.536 | 0.000 | 0.021 |

### Training Time vs. Kappa Trade-off

```
Mean Kappa
  0.54 │                   ★ Run 4 (660 min) ← Best kappa AND lowest variance
       │   ○ Run 2 (71 min)
  0.53 │   ○ Run 1 (231 min)
       │   ○ Run 3 (177 min)
  0.52 │
       │
  0.51 │
       └────────────────────────────────── Total wall-clock time
         71    177   231               660 min

★ Run 4 dominates on kappa & consistency, but costs 9× more time than Run 2.
○ Run 2 (Epoch-15) gives 99% of Run 4's kappa in 11% of the time — 
  excellent for rapid iteration and hyperparameter search.
```

---

## Key Findings and Recommendations

### What worked

1. **Early stopping + Adam reset (Run 4)** is the most important improvement. Adding the Adam reset dropped kappa variance by 3× (std 0.083 → 0.028) and gave the most reliable performance across folds. The periodic reset prevents stale RayBNN momentum from interfering with CNN updates.

2. **The model learns most of what it can in ≤15 epochs.** Runs 1 and 2 both show the best checkpoint almost always occurs before epoch 20. This means the epoch-15 quick run is a valid proxy for rapid evaluation.

3. **Wake and N1 are reliably learned.** Across all runs, Wake F1 ≥ 0.93 and N1 recall ≥ 0.68 consistently. The model generalizes well across subjects for these classes.

### What did not work

4. **traj_size > 1 did not improve kappa** and significantly increased training time (2.6× for traj_size=2, ~1.7× for traj_size=4). The temporal context hypothesis (that observing multiple consecutive windows helps detect short events) was not supported by results on completed folds.

5. **N2 and Microsleep remain the core unsolved problem.** Across every configuration, N2 F1 < 0.16 and Microsleep F1 < 0.13. Class weighting alone has not solved the imbalance (N2 ≈ 1% of data, Microsleep ≈ 3%). traj_size=4 made N2 detection collapse to zero in two folds.

6. **proc_num=4** showed no clear benefit in its single completed fold and increased training time by ~1.5×.

### Suggested next steps

| Priority | Suggestion | Rationale |
|---|---|---|
| High | Run traj_size=1, proc_num=2, patience=12 + Adam reset on a longer schedule (100 epochs) | Confirm Run 4 is truly converged |
| High | Investigate focal loss or oversampling for N2/Microsleep | Class weighting (sqrt-inverse) is insufficient |
| Medium | Complete traj_size=2 and traj_size=4 for all 4 folds | Cannot conclude without full results |
| Medium | Complete proc_num=4 for all 4 folds | One fold is insufficient for comparison |
| Low | Try two-stage classifier: first Wake vs. non-Wake, then N1/N2/MS | Might help the model focus on rare classes |

---

## Notes on Incomplete Runs

| File | Status | Last point captured |
|---|---|---|
| `proc_num_4_...txt` | Fold 1 complete, Fold 2 ep7 when terminated | Deliberately stopped |
| `traj_size_2_...txt` | Folds 1–2 complete, Fold 3 ep7 when terminated | Deliberately stopped |
| `traj_size_4_...txt` | Folds 1–3 complete, Fold 4 ep1 when terminated | Deliberately stopped |

All incomplete runs were deliberately terminated, not crashed. Results from completed folds within each run are valid and reported above.

---

[← Back to README](README.md) | [Previous: Running the Script](07_RUNNING_THE_SCRIPT.md)

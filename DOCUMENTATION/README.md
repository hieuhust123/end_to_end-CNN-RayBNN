# MWT EEG Classification: CNN + RayBNN Documentation

**For**: ML engineers new to this codebase  
**Purpose**: Comprehensive reference guide for understanding, running, and maintaining the end-to-end deep learning pipeline  
**Target Audience**: Junior ML engineers, incoming researchers (comfortable with PyTorch basics)

---

## 📋 Quick Navigation

This documentation is organized into modular guides. Start with **Overview**, then jump to whatever section is most relevant:

| Section | Focus | Best For |
|---------|-------|----------|
| **[1. Overview](01_OVERVIEW.md)** | What does this script do? High-level architecture. | Getting oriented |
| **[2. Architecture](02_ARCHITECTURE.md)** | CNN + RayBNN design, layer breakdown, integration. | Understanding the model |
| **[3. Data Pipeline](03_DATA_PIPELINE.md)** | EEG dataset, windowing, trajectory sampling, batching. | Understanding data flow |
| **[4. Model Components](04_MODEL_COMPONENTS.md)** | CNN implementation details, custom autograd, gradient computation. | Deep technical dive |
| **[5. Training Flow](05_TRAINING_FLOW.md)** | Training loop, optimizer, deferred BNN updates, loss tracking. | Understanding training |
| **[6. Evaluation](06_EVALUATION.md)** | Cross-validation, metrics, diagnostic blocks, interpreting results. | Interpreting experiments |
| **[7. Running the Script](07_RUNNING_THE_SCRIPT.md)** | Installation, setup, running end-to-end, output interpretation. | Actually running it |
| **[8. Experiment Results](08_EXPERIMENT_RESULTS.md)** | Summary of all experimental runs: configs, per-fold kappa, per-class breakdown, findings. | Reviewing experiment outcomes |
| **[9. Data Pipeline Summary](09_DATA_PIPELINE_SUMMARY.md)** | Concise key-points summary of the data pipeline with diagrams: flow, windowing, splits, batching. | Quick reference / presenting the pipeline |

---

## 🎯 Common Starting Points

**"I need to get this running immediately"**  
→ Jump to [Running the Script](07_RUNNING_THE_SCRIPT.md)

**"I need to understand what this code does"**  
→ Start with [Overview](01_OVERVIEW.md), then [Architecture](02_ARCHITECTURE.md)

**"I'm debugging training issues"**  
→ Read [Training Flow](05_TRAINING_FLOW.md) and [Model Components](04_MODEL_COMPONENTS.md)

**"I need to evaluate results or tune hyperparameters"**  
→ See [Evaluation](06_EVALUATION.md)

**"I want to see what experiments have been run and what worked"**  
→ See [Experiment Results](08_EXPERIMENT_RESULTS.md)

**"I want a quick-reference summary of the data pipeline with diagrams"**  
→ See [Data Pipeline Summary](09_DATA_PIPELINE_SUMMARY.md)

**"I need the full technical breakdown"**  
→ Read in order: Overview → Architecture → Data Pipeline → Model Components → Training Flow → Evaluation

---

## 🔑 Key Concepts at a Glance

| Term | Meaning |
|------|---------|
| **MWT EEG** | Maintenance of Wakefulness Test EEG data; 76 subjects with sleep stage labels |
| **CNN_EEG** | 12-layer convolutional neural network mimicking Xuan Chen's Keras implementation |
| **RayBNN** | Ray Bayesian Neural Network; a Rust library providing Bayesian classification layer |
| **Trajectory** | Overlapping 16-second EEG windows sampled 4 seconds apart (traj_size, traj_stride) |
| **AutoGrad** | Custom PyTorch autograd function bridging CNN gradients with RayBNN Rust code |
| **4-Fold CV** | Subject-independent cross-validation: 4 folds, each fold ~19 subjects test, ~57 subjects train |
| **Kappa** | Cohen's kappa statistic; measures agreement accounting for chance (target metric) |

---

## 📊 What the Script Does (TL;DR)

1. **Load** MWT EEG data (2 channels: O1, O2) from MATLAB files
2. **Preprocess** signals: pad, normalize, create overlapping 16-second windows
3. **Extract features** using a 12-layer CNN (outputs 256-dim vectors)
4. **Classify** using RayBNN (Bayesian 256→4 class predictor)
5. **Train jointly** CNN + RayBNN with custom gradient flow
6. **Evaluate** via 4-fold cross-validation, report accuracy/kappa/F1

Output: 4 trained models (one per fold), performance metrics, training plots.

---

## ⚙️ Tech Stack

- **Python 3.8+** 
- **PyTorch** — deep learning framework
- **RayBNN (Rust)** — Bayesian classification layer (via Python bindings)
- **scikit-learn** — metrics (kappa, F1, precision, recall)
- **NumPy/SciPy** — numerical computing, MATLAB file loading
- **CUDA** — GPU acceleration (if available)

---

## 📁 Repository Structure

```
RayBNN_Python/
├── Python_Code/
│   └── mwt_test_backward_cnn+raybnn_testing.py  ← Main script (this is what we document)
├── mwt_eeg/                      ← MWT EEG data directory (.mat files)
├── DOCUMENTATION/                ← You are here
│   ├── README.md                 (this file)
│   ├── 01_OVERVIEW.md
│   ├── 02_ARCHITECTURE.md
│   ├── 03_DATA_PIPELINE.md
│   ├── 04_MODEL_COMPONENTS.md
│   ├── 05_TRAINING_FLOW.md
│   ├── 06_EVALUATION.md
│   ├── 07_RUNNING_THE_SCRIPT.md
│   ├── 08_EXPERIMENT_RESULTS.md
│   └── 09_DATA_PIPELINE_SUMMARY.md
└── raybnn/                       ← RayBNN library code (Rust + Python bindings)
```

---

## 🚀 How to Use This Documentation

1. **First time here?** Read [01_OVERVIEW.md](01_OVERVIEW.md) fully.
2. **Want to run it?** Skip to [07_RUNNING_THE_SCRIPT.md](07_RUNNING_THE_SCRIPT.md).
3. **Have errors?** Check the relevant section (e.g., training issues → [05_TRAINING_FLOW.md](05_TRAINING_FLOW.md)).
4. **Need to modify code?** Read [04_MODEL_COMPONENTS.md](04_MODEL_COMPONENTS.md) and [05_TRAINING_FLOW.md](05_TRAINING_FLOW.md).
5. **Interpreting results?** See [06_EVALUATION.md](06_EVALUATION.md).

---

## 📝 Document Metadata

- **Last Updated**: 2026-04-16
- **Script Version**: Uses PyTorch + RayBNN bindings
- **Documentation Scope**: Covers sections 1-1600 of `mwt_test_backward_cnn+raybnn_testing.py`
- **Intended Audience**: ML juniors with PyTorch familiarity

---

## 💡 Quick FAQ

**Q: Do I need CUDA to run this?**  
A: No, but it will be very slow on CPU. CUDA/GPU is strongly recommended.

**Q: What's the expected runtime?**  
A: ~20-30 minutes per fold on a modern GPU (4 folds = 1.5-2 hours total).

**Q: Can I use different batch sizes?**  
A: Yes, but RayBNN has a fixed batch size architecture. See [Running the Script](07_RUNNING_THE_SCRIPT.md).

**Q: What is RayBNN?**  
A: A Bayesian neural network library (Rust-based). We treat it as a black box. See [Overview](01_OVERVIEW.md) for more.

**Q: How do I know if training is working?**  
A: Check the diagnostic outputs (AREA 1-4) in the first batch/epoch. See [Training Flow](05_TRAINING_FLOW.md).

---

## 🔗 See Also

- Original Xuan Chen implementation: `Xuan_Chen_code/c/`
- RayBNN Rust source: `raybnn/`
- Example data: `mwt_eeg/*.mat`

---

**Next**: Start with [01_OVERVIEW.md](01_OVERVIEW.md) →

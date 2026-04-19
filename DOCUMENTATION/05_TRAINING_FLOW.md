# 5. Training Flow: How Models Learn

[← Back to README](README.md)

## Training Architecture

```
For each fold (1 of 4):
  ├─ Initialize CNN + RayBNN
  ├─ Create train/test DataLoaders
  ├─ Set optimizer (Adam) + scheduler (CosineAnnealing) + loss (weighted CE)
  │
  └─ For each epoch (0 to 49):
      │
      ├─ Training phase:
      │   ├─ For each batch (1000 samples) in train_loader:
      │   │   ├─ Forward: batch_x → CNN → features → RayBNN → logits
      │   │   ├─ Loss: CrossEntropyLoss(logits, batch_y, weight=class_weights)
      │   │   ├─ Backward: AutoGradEndtoEnd.backward() + CNN.backward()
      │   │   ├─ Optimizer.step(): Update CNN params
      │   │   └─ Deferred BNN update: Update RayBNN params
      │   │
      │   └─ Log: epoch loss, train accuracy
      │
      ├─ Evaluation phase (every 2 epochs):
      │   ├─ For each batch in test_loader:
      │   │   ├─ Inference: batch_x → logits (no gradients)
      │   │   ├─ Loss: unweighted CE
      │   │   └─ Predictions: argmax(logits)
      │   │
      │   └─ Compute metrics: accuracy, kappa, F1
      │
      ├─ Early stopping check:
      │   ├─ If test_kappa > best_kappa:
      │   │   ├─ Save best_kappa, best_epoch
      │   │   └─ Checkpoint CNN weights
      │   ├─ Else:
      │   │   └─ Increment stale_evals counter
      │   │
      │   └─ If stale_evals ≥ 12 (patience): Stop & restore best checkpoint
      │
      └─ LR scheduler.step(): Decay learning rate

After all epochs:
  └─ Final evaluation: Compute metrics on full test set
```

---

## Training Loop Code

```python
def train_ete_model(model, train_dataset, test_dataset, alpha0, batch_size, 
                    max_epoch=50, mode="both", eval_every=2, fold_idx=0):
    """
    Args:
        model: EndtoEndTrainer instance (CNN + RayBNN)
        mode: "both" (train both), "cnn_only", "raybnn_only", or "frozen"
        eval_every: Evaluate every N epochs (e.g., eval_every=2 → epochs 0,2,4,...)
        fold_idx: Which fold (0-3); controls diagnostic verbosity
    """
    
    device = model.device
    
    # ──────── Create DataLoaders ────────
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,  # 1000 (RayBNN fixed)
        shuffle=True,           # Randomize each epoch
        num_workers=4,          # Parallel loading
        pin_memory=True,        # GPU memory pre-allocation
        drop_last=True,         # Drop < batch_size final batch
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # No shuffle for evaluation
        num_workers=4,
        pin_memory=True,
        drop_last=False         # Keep all test samples
    )
    
    # ──────── Loss Function ────────
    # Compute class weights (inverse frequency)
    all_labels = np.concatenate(train_dataset.labels)
    label_counts = np.bincount(all_labels, minlength=4).astype(np.float32)
    label_counts = np.maximum(label_counts, 1.0)
    class_weights = 1.0 / np.sqrt(label_counts)  # sqrt-inverse
    class_weights = class_weights / class_weights.sum() * 4  # normalize
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Training: weighted loss (balances classes)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Evaluation: unweighted loss (for fair comparison across runs)
    criterion_eval = torch.nn.CrossEntropyLoss()
    
    # ──────── Optimizer & Scheduler ────────
    # Determine learning rate based on mode
    if mode == "cnn_only":
        lr = 0.0003
        alpha0 = 0.0  # Freeze RayBNN
    elif mode == "raybnn_only":
        lr = 0.0  # Freeze CNN
    elif mode == "frozen":
        lr = 0.0
        alpha0 = 0.0  # Freeze both
    else:  # "both"
        lr = 0.0003
        # alpha0 stays as passed in (e.g., 0.0005)
    
    # Create optimizer only if CNN learning rate > 0
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4  # L2 regularization: penalizes large weights
    ) if lr > 0 else None
    
    # Cosine annealing: lr decays smoothly from lr → lr*0.01
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epoch,
        eta_min=lr * 0.01
    ) if optimizer is not None else None
    
    # ──────── Training Setup ────────
    model.train()  # Set to training mode (enables dropout, Gaussian noise)
    
    # Early stopping state
    best_kappa = -np.inf
    best_kappa_epoch = 0
    evals_since_best = 0
    early_stop_patience = 12  # Evaluation checks without improvement
    best_model_state = None
    
    # Accumulators for final summary
    all_epoch_losses = []
    all_epoch_accs = []
    all_test_losses = []
    all_test_accs = []
    all_test_kappas = []
    
    # ─────────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────────
    
    for epoch in range(max_epoch):
        model.update_epoch(epoch)  # Inform autograd of current epoch
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start_time = time.perf_counter()
        
        # ──────── TRAINING PHASE ────────
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # For traj_size > 1: batch_y is (batch, traj_size)
            # Use slot 0 for loss (all slots have identical label)
            batch_y_loss = batch_y[:, 0] if batch_y.dim() == 2 else batch_y
            
            # Zero gradients from previous iteration
            if optimizer:
                optimizer.zero_grad()
            
            # Update batch counter (for diagnostics)
            model.update_batch(batch_idx)
            
            # ──── FORWARD PASS ────
            output = model(batch_x, batch_y, verbose=False)
            # output: (batch, 4) logits
            
            # ──── LOSS COMPUTATION ────
            loss = criterion(output, batch_y_loss)
            
            # Detect loss explosions (sanity check)
            if loss.item() > 10.0 or torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  LOSS EXPLOSION: {loss.item():.3f} at epoch {epoch+1}, batch {batch_idx}")
                print("Stopping training to prevent divergence")
                return model
            
            # ──── BACKWARD PASS ────
            if mode in ("both", "cnn_only"):
                loss.backward()  # Computes gradients via AutoGradEndtoEnd + CNN
                # After backward, gradients are in:
                # - CNN parameter.grad tensors (from PyTorch autograd)
                # - AutoGradEndtoEnd._pending_bnn_grad (from Rust)
            
            # ──── OPTIMIZER STEP ────
            if optimizer:
                optimizer.step()  # Updates CNN weights from collected gradients
            
            # ──── DEFERRED BNN UPDATE ────
            # Apply RayBNN update AFTER CNN optimizer.step()
            # This ensures both see the same forward state
            if mode in ("both", "raybnn_only"):
                apply_deferred_bnn_update(alpha0, model._autograd_cls)
            
            # ──── METRICS ACCUMULATION ────
            epoch_loss += loss.item()
            with torch.no_grad():
                predictions = output.argmax(dim=1)  # (batch,) class indices
                epoch_total += batch_y_loss.size(0)
                epoch_correct += (predictions == batch_y_loss).sum().item()
        
        # ──────── END-OF-EPOCH SUMMARY ────────
        epoch_time = time.perf_counter() - epoch_start_time
        final_epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        all_epoch_losses.append(avg_epoch_loss)
        all_epoch_accs.append(final_epoch_acc)
        
        # ──────── LEARNING RATE SCHEDULING ────────
        if scheduler is not None:
            scheduler.step()  # Decay LR for next epoch
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr
        
        # ──────── EVALUATION PHASE (every eval_every epochs) ────────
        run_eval = (epoch % eval_every == 0) or (epoch == max_epoch - 1)
        
        if run_eval:
            model.eval()  # Disable dropout, Gaussian noise
            test_loss_sum = 0.0
            test_correct = 0
            test_total = 0
            all_preds_epoch = []
            all_labels_epoch = []
            
            with torch.no_grad():
                for test_batch_idx, (test_x, test_y) in enumerate(test_loader):
                    test_x = test_x.to(device, non_blocking=True)
                    test_y = test_y.to(device, non_blocking=True)
                    test_y_loss = test_y[:, 0] if test_y.dim() == 2 else test_y
                    
                    # Pad if needed (RayBNN requires fixed batch_size)
                    actual_batch = test_x.size(0)
                    if actual_batch < batch_size:
                        # Pad to batch_size, pad labels with zeros
                        pad_x = torch.zeros(batch_size, *test_x.shape[1:], device=device)
                        pad_y = torch.zeros(batch_size, *test_y.shape[1:], 
                                           dtype=test_y.dtype, device=device)
                        pad_x[:actual_batch] = test_x
                        pad_y[:actual_batch] = test_y
                        node.update_batch(test_batch_idx)
                        test_output = model(pad_x, pad_y, verbose=False)
                        test_output = test_output[:actual_batch]  # Discard padding predictions
                    else:
                        model.update_batch(test_batch_idx)
                        test_output = model(test_x, test_y, verbose=False)
                    
                    test_loss_sum += criterion_eval(test_output, test_y_loss).item()
                    test_correct += (test_output.argmax(dim=1) == test_y_loss).sum().item()
                    test_total += actual_batch
                    
                    all_preds_epoch.append(test_output.argmax(dim=1).cpu().numpy())
                    all_labels_epoch.append(test_y_loss.cpu().numpy())
            
            # Compute metrics
            all_preds_epoch = np.concatenate(all_preds_epoch)
            all_labels_epoch = np.concatenate(all_labels_epoch)
            test_kappa = cohen_kappa_score(all_labels_epoch, all_preds_epoch)
            
            avg_test_loss = test_loss_sum / len(test_loader)
            test_acc = test_correct / test_total if test_total > 0 else 0.0
            
            model.train()  # Back to training mode
        else:
            # Carry forward previous test metrics (since we didn't evaluate)
            avg_test_loss = all_test_losses[-1] if all_test_losses else 0.0
            test_acc = all_test_accs[-1] if all_test_accs else 0.0
            test_kappa = all_test_kappas[-1] if all_test_kappas else 0.0
        
        all_test_losses.append(avg_test_loss)
        all_test_accs.append(test_acc)
        all_test_kappas.append(test_kappa)
        
        # ──────── EARLY STOPPING ────────
        if run_eval:
            if test_kappa > best_kappa:
                best_kappa = test_kappa
                best_kappa_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.cnn.state_dict())
                evals_since_best = 0
            else:
                evals_since_best += 1
        
        # ──────── LOGGING ────────
        eval_tag = "" if run_eval else " (cached)"
        early_stop_tag = f"[best={best_kappa:.4f}@ep{best_kappa_epoch}, stale={evals_since_best}/{early_stop_patience}]"
        
        print(f"Epoch {epoch+1}/{max_epoch} | "
              f"loss={avg_epoch_loss:.4f} acc={final_epoch_acc:.4f} | "
              f"test_loss={avg_test_loss:.4f}{eval_tag} test_acc={test_acc:.4f}{eval_tag} "
              f"kappa={test_kappa:.4f}{eval_tag} | "
              f"lr={current_lr:.6f} {early_stop_tag}")
        
        # ──────── STOPPING CRITERIA ────────
        if evals_since_best >= early_stop_patience:
            print(f"\n⚠️  Early stopping: No improvement for {early_stop_patience} evaluations.")
            print(f"Best kappa: {best_kappa:.4f} at epoch {best_kappa_epoch}")
            
            # Restore best weights
            if best_model_state is not None:
                model.cnn.load_state_dict(best_model_state)
                print("CNN restored to best checkpoint.")
            break
        
        # ──────── ADAM STATE RESET (every 15 epochs) ────────
        # Prevents gradient collapse from stale momentum
        if (epoch + 1) % 15 == 0 and epoch + 1 < max_epoch:
            model._autograd_cls._adam_mt = None
            model._autograd_cls._adam_vt = None
            print(f"  [Reset] RayBNN Adam state at epoch {epoch+1}")
    
    # ──────── FINAL SUMMARY ────────
    print(f"\n{'='*70}")
    print(f"Final train accuracy: {all_epoch_accs[-1]:.4f}")
    print(f"Final test accuracy: {all_test_accs[-1]:.4f}")
    print(f"Final test kappa: {all_test_kappas[-1]:.4f}")
    print(f"Best test kappa: {max(all_test_kappas):.4f} (@epoch {all_test_kappas.index(max(all_test_kappas))+1})")
    print(f"{'='*70}")
    
    return model, {
        "epoch_losses": all_epoch_losses,
        "epoch_accs": all_epoch_accs,
        "test_losses": all_test_losses,
        "test_accs": all_test_accs,
        "test_kappas": all_test_kappas,
    }
```

---

## Learning Rate Schedule

### Why Cosine Annealing?

```
Learning Rate vs Epoch:
     
lr
│     ╭─ lr (initial, e.g., 0.0003)
│    ╱
│   ╱    ← Cosine decay curve
│  ╱
│ ╱
└──────────────────────────────── epoch
 0        15       30       45      50 (max_epoch)

At epoch 0: lr = 0.0003 (fast updates, explore parameter space)
At epoch 25: lr ≈ 0.00015 (medium, fine-tuning)
At epoch 49: lr = 0.000003 (slow, polishing)

Benefit:
• Early epochs: High LR, rapid descent into loss minimum
• Late epochs: Low LR, fine-tune without overshooting
• No hard cutoff: smooth decay helps convergence
```

### Implementation

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epoch,        # Full decay over max_epoch epochs
    eta_min=lr * 0.01       # Minimum LR (1% of initial)
)

# Each epoch:
scheduler.step()
current_lr = scheduler.get_last_lr()[0]  # Get current LR for logging
```

---

## Early Stopping

### Why Early Stopping?

```
Train Accuracy ← Keeps improving
                  ╱  ╱  ╱
                ╱  ╱
              ╱

Test Accuracy ← Saturates, then declines (overfitting!)
              ╱
           ▂▃▃▃░░░░░░
          ╱          ╲  ← Model fits noise in training data
                        doesn't generalize to test

           Ideal stop point
           ↑
           epoch 30 (before decline starts)
```

### Implementation

```python
best_kappa = -np.inf
best_kappa_epoch = 0
evals_since_best = 0
early_stop_patience = 12  # Stop after 12 eval checks without improvement
best_model_state = None

for epoch in range(max_epoch):
    # ... training ...
    
    if run_eval:
        # Compute test kappa
        test_kappa = cohen_kappa_score(all_preds, all_labels)
        
        if test_kappa > best_kappa:
            # New best!
            best_kappa = test_kappa
            best_kappa_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.cnn.state_dict())
            evals_since_best = 0
        else:
            # No improvement
            evals_since_best += 1
    
    # Check stopping
    if evals_since_best >= early_stop_patience:
        print(f"Early stop: no improvement for {early_stop_patience} evaluations")
        # Restore best weights before returning
        if best_model_state:
            model.cnn.load_state_dict(best_model_state)
        break
```

---

## Class Weighting in Loss

### Effect on Gradient

```python
# Unweighted loss:
loss = sum(CE(logit[i], label[i]))  # Same weight for all samples
∂loss/∂param ∝ [all_samples equally]

# Weighted loss:
class_weight = [1.0, 1.05, 1.30, 1.83]  # Wake, N1, N2, Microsleep
loss = sum(class_weight[label[i]] * CE(logit[i], label[i]))
∂loss/∂param ∝ [minority samples weighted 3x higher!]

Result: CNN learns to distinguish rare Microsleep events
        instead of just predicting Wake always.
```

### Sqrt vs Direct Inverse

```python
Without weighting: [431k, 12k, 8k, 4k] labels
                    Wake  N1   N2   MS

Direct inverse (1/count):
  [0.0000023, 0.000083, 0.000125, 0.00025]
  Microsleep is 100× more weighted than Wake!
  → Model obsesses over tiny MS class, overfit.

Sqrt-inverse (1/√count):
  [0.0048, 0.009, 0.011, 0.016]
  Microsleep is ~3.3× more weighted than Wake
  → Balanced, reasonable.

Final normalization (× 4 / sum):
  [0.55, 1.05, 1.30, 1.83]  ← Final weights used
```

---

## Deferred vs Immediate BNN Updates

### The Move-Target Problem

**Bad (old approach)**:
```
Iteration t:
  1. Forward: CNN(X) → features → RayBNN(features) → loss
  2. Backward: grad_RayBNN, update RayBNN params
  3. Backward: grad_CNN
  4. Optimizer.step(): update CNN params

Problem: CNN gradients computed now, but RayBNN params changed in step 2!
Result: CNN updates based on OLD RayBNN, loose coupling
```

**Good (new approach)**:
```
Iteration t:
  1. Forward: CNN(X) → features → RayBNN(V_old) → loss
  2. Backward: grad_RayBNN, STORE it
  3. Backward: grad_CNN
  4. Optimizer.step(): update CNN
  5. Apply deferred update: update RayBNN

Benefit: CNN and RayBNN both optimize against SAME forward state
Result: Tight coupling, stable joint training
```

---

## Next Steps

- **Understand evaluation & results**: [Evaluation](06_EVALUATION.md)
- **Run the script**: [Running the Script](07_RUNNING_THE_SCRIPT.md)
- **Debug training issues**: [Training Flow](05_TRAINING_FLOW.md) (this page)

---

[← Back to README](README.md) | [Previous: Model Components](04_MODEL_COMPONENTS.md) | [Next: Evaluation →](06_EVALUATION.md)

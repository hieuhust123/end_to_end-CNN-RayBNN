



import raybnn_python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import os
import scipy.io as spio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ==================== MWT EEG Dataset (On-the-fly windowing) ====================

class MWTDataset(torch.utils.data.Dataset):
    """MWT EEG dataset that windows raw signals on-the-fly.

    Stores only raw padded signals in RAM (~280 MB for 76 subjects).
    Each __getitem__ call extracts a single 16s window from the raw signal.
    """

    def __init__(self, mat_dir, file_list, w_len=1600, stride=1):
        """
        Args:
            mat_dir: path to directory containing .mat files
            file_list: list of .mat filenames to include
            w_len: half-window size in samples (1600 = 8s at 200Hz, full window = 16s)
            stride: take every Nth sample (stride=16 reduces 28.8M to ~1.8M with negligible info loss)
        """
        self.w_len = w_len
        self.data_dim = w_len * 2  # 3200 samples = 16 seconds
        self.stride = stride

        # Load raw signals and pad them
        self.signals_O1 = []  # list of (483200,) arrays
        self.signals_O2 = []
        self.labels = []      # list of (480000,) arrays
        self.subject_n_samples = []  # number of strided samples per subject
        self.sample_offsets = []  # cumulative strided sample count for index mapping

        total_samples = 0
        for fname in file_list:
            mat = spio.loadmat(os.path.join(mat_dir, fname), struct_as_record=False, squeeze_me=True)
            data = mat['Data']

            eeg_O1 = data.eeg_O1.astype(np.float32)
            eeg_O2 = data.eeg_O2.astype(np.float32)
            labels = data.labels_O1.astype(np.int64)

            # Pad signals by w_len on both sides
            eeg_O1_padded = np.pad(eeg_O1, (w_len, w_len), mode='constant', constant_values=0)
            eeg_O2_padded = np.pad(eeg_O2, (w_len, w_len), mode='constant', constant_values=0)

            # Pre-normalize: divide by 100, clip to [-1, 1] (avoids per-sample work in __getitem__)
            eeg_O1_padded = np.clip(eeg_O1_padded / 100.0, -1.0, 1.0)
            eeg_O2_padded = np.clip(eeg_O2_padded / 100.0, -1.0, 1.0)

            self.signals_O1.append(eeg_O1_padded)
            self.signals_O2.append(eeg_O2_padded)
            self.labels.append(labels)

            # Number of samples after striding for this subject
            n_raw = len(labels)
            n_strided = (n_raw + stride - 1) // stride  # ceil division
            self.subject_n_samples.append(n_strided)
            self.sample_offsets.append(total_samples)
            total_samples += n_strided

        self.total_samples = total_samples
        self.sample_offsets.append(total_samples)  # sentinel

        # Build subject boundaries for fast index lookup
        self.sample_offsets = np.array(self.sample_offsets, dtype=np.int64)

        raw_total = sum(len(l) for l in self.labels)
        print(f"MWTDataset: {len(file_list)} subjects, {raw_total:,} raw samples, "
              f"stride={stride} -> {self.total_samples:,} effective samples")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which subject this sample belongs to (binary search)
        subj_idx = np.searchsorted(self.sample_offsets[1:], idx, side='right')
        strided_idx = idx - self.sample_offsets[subj_idx]
        # Map strided index back to raw sample index
        sample_idx = int(strided_idx * self.stride)
        # Clamp to valid range
        max_idx = len(self.labels[subj_idx]) - 1
        sample_idx = min(sample_idx, max_idx)

        # Extract 16s window centered on this sample
        # After padding, sample_idx in the padded signal starts at sample_idx (not sample_idx + w_len)
        # because the label at position sample_idx corresponds to the window centered at sample_idx + w_len
        center = sample_idx + self.w_len
        start = center - self.w_len
        end = center + self.w_len  # end = start + 3200

        window_O1 = self.signals_O1[subj_idx][start:end]  # (3200,)
        window_O2 = self.signals_O2[subj_idx][start:end]  # (3200,)

        # Already pre-normalized in __init__, just stack and reshape for Conv2D: (2, 3200, 1)
        window = np.stack([window_O1, window_O2], axis=0)  # (2, 3200)
        window = window[:, :, np.newaxis]  # (2, 3200, 1)

        label = self.labels[subj_idx][sample_idx]

        return torch.from_numpy(window).float(), torch.tensor(label, dtype=torch.long)


def split_subjects(mat_dir, n_train=60, seed=42):
    """Split 76 subjects into train/test sets."""
    all_files = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat')])
    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    train_files = all_files[:n_train]
    test_files = all_files[n_train:]
    print(f"Train subjects: {len(train_files)}, Test subjects: {len(test_files)}")
    return train_files, test_files


# ==================== CNN for EEG (simplified, mirrors MNIST CNN structure) ====================

class CNN_EEG(nn.Module):
    """Simple Conv2D CNN for EEG signals (3 conv layers, same structure as MNIST CNN).

    Input:  (batch, 2, 3200, 1) -- 2 EEG channels, 3200 time steps, width=1
    Output: (batch, 256) -- feature vector

    Architecture (mirrors MNIST CNN):
        Conv2d(2->12,  (3,1)) + ReLU + MaxPool(4,1)     3200 -> 800
        Conv2d(12->64, (3,1)) + ReLU + MaxPool(4,1)      800 -> 200
        Conv2d(64->16, (3,1)) + ReLU + Dropout
        AdaptiveAvgPool2d((16,1))                         200 -> 16
        Flatten -> 16 * 16 * 1 = 256
    """

    def __init__(self):
        super(CNN_EEG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=12,
            kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool = nn.MaxPool2d(kernel_size=(4, 1))
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=64,
            kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=16,
            kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.drop = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 1))

    def forward(self, x, y_labels, verbose=False):
        """
        Args:
            x: (batch, 2, 3200, 1) EEG windows
            y_labels: not used in forward, kept for API compatibility
            verbose: print shape diagnostics
        """
        # First conv + pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        if verbose:
            print(f"After conv1 + ReLU + pool: {x.shape}")  # (batch, 12, 800, 1)

        # Second conv + pool
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if verbose:
            print(f"After conv2 + ReLU + pool: {x.shape}")  # (batch, 64, 200, 1)

        # Third conv + dropout
        x = F.relu(self.conv3(x))
        if verbose:
            print(f"After conv3 + ReLU: {x.shape}")  # (batch, 16, 200, 1)

        x = self.drop(x)

        # Adaptive pool to fixed size, then flatten -> 256
        x = self.adaptive_pool(x)
        if verbose:
            print(f"After adaptive pool: {x.shape}")  # (batch, 16, 16, 1)

        features_flat = x.reshape(x.size(0), -1)
        if verbose:
            print(f"After flatten: {features_flat.shape}")  # (batch, 256)

        return features_flat


# ==================== Kappa metric (from Xuan Chen) ====================

def kappa_metric(y_true, y_pred, n_cl=4):
    """Computes Cohen kappa per class."""
    y = np.array(y_true)
    y_ = np.array(y_pred)
    res = []
    for c in range(n_cl):
        res.append(cohen_kappa_score(y == c, y_ == c))
    return np.array(res)


# ==================== main() -- setup model + data ====================

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ## Load MWT EEG dataset
    mat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mwt_eeg')
    mat_dir = os.path.normpath(mat_dir)
    print(f"MWT EEG data directory: {mat_dir}")

    train_files, test_files = split_subjects(mat_dir, n_train=60, seed=42)

    print("Loading training dataset...")
    train_dataset = MWTDataset(mat_dir, train_files, stride=16)
    print("Loading test dataset...")
    test_dataset = MWTDataset(mat_dir, test_files, stride=16)

    ## Parameter setting for MWT EEG dataset
    dir_path = "/tmp/"
    max_input_size = 256
    input_size = 256

    max_output_size = 4
    output_size = 4

    max_neuron_size = 2000

    batch_size = 1000
    traj_size = 1

    proc_num = 2
    active_size = 1000

    training_samples = 1  # Each AutoGrad call processes 1 batch (batch_size samples)
    crossval_samples = 1
    testing_samples = 1

    alpha0 = 0.0005

    #Create Neural Network
    arch_search = raybnn_python.create_start_archtecture(
        input_size,
        max_input_size,

        output_size,
        max_output_size,

        active_size,
        max_neuron_size,

        batch_size,
        traj_size,

        proc_num,
        dir_path
    )

    sphere_rad = arch_search["neural_network"]["netdata"]["sphere_rad"]

    arch_search = raybnn_python.add_neuron_to_existing3(
        10,
        10000,
        sphere_rad/1.3,
        sphere_rad/1.3,
        sphere_rad/1.3,

        arch_search,
    )

    arch_search = raybnn_python.select_forward_sphere(arch_search)

    raybnn_python.print_model_info(arch_search)

    max_epoch = 50

    class AutoGradEndtoEnd(torch.autograd.Function):

        _current_epoch = 0
        _current_batch = 0
        _arch_search = None  # Will be set after class definition
        _pending_bnn_grad = None  # Raw negated BNN gradient (deferred update)
        _pending_bnn_params = None  # Current BNN params before update
        _adam_mt = None  # Adam first moment
        _adam_vt = None  # Adam second moment
        @staticmethod
        def forward(ctx, features_flat, y_labels, batch_size,
        traj_size, max_epoch, input_size, output_size, training_samples, alpha0):


            # Save tensors that will be needed in backward
            ctx.batch_size = batch_size
            ctx.traj_size = traj_size
            ctx.max_epoch = max_epoch
            ctx.input_size = input_size
            ctx.output_size = output_size
            ctx.training_samples = training_samples
            ctx.alpha0 = alpha0

            # features_flat.shape # torch.Size([1000, 256])
            # y_labels.shape # torch.Size([1000])

            # Convert X and Y to numpy arrays (Convert PyTorch -> NumPy)
            features_np = features_flat.detach().cpu().numpy()
            y_labels_np = y_labels.detach().cpu().numpy()

            # Create training arrays with 1 trajectory (this call = 1 batch)
            train_x = np.zeros((input_size, batch_size, traj_size, 1), dtype=np.float32)
            train_y = np.zeros((output_size, batch_size, traj_size, 1), dtype=np.float32)

            # Vectorized data formatting (replaces slow per-sample loop)
            train_x[:, :, 0, 0] = features_np.T  # (256, 1000)
            indices = y_labels_np.astype(int)
            valid = (indices >= 0) & (indices < output_size)
            train_y[indices[valid], np.where(valid)[0], 0, 0] = 1.0

            _verbose = (AutoGradEndtoEnd._current_batch == 0)
            if _verbose:
                print("[FORWARD AutoGrad] CNN features reshaped as RayBNN input: ", train_x.shape)

            # Forward returns only Yhat (single numpy array). arch_search is not modified.
            Yhat_array = raybnn_python.state_space_forward_batch(train_x, train_y,
            traj_size, max_epoch, AutoGradEndtoEnd._arch_search)

            # Convert to float32
            Yhat_array = np.array(Yhat_array).astype(np.float32)

            ctx.save_for_backward(features_flat, y_labels)
            # Cache formatted arrays so backward() doesn't rebuild them
            ctx.train_x = train_x
            ctx.train_y = train_y

            # Convert Yhat from numpy arrays to Pytorch tensors
            Yhat_tensor = torch.from_numpy(Yhat_array).to(features_flat.device)

            # Yhat_tensor.shape # torch.Size([4, 1000, 1, 1])
            Yhat = Yhat_tensor.squeeze(-1).squeeze(-1).T

            if _verbose:
                print(f"[MAIN FWD] Yhat mean={Yhat.mean().item():.6f} max={Yhat.max().item():.6f} min={Yhat.min().item():.6f} std={Yhat.std().item():.6f}")

            return Yhat

        @staticmethod
        def backward(ctx, grad_output):
            features_flat, y_labels = ctx.saved_tensors
            batch_size = ctx.batch_size
            traj_size = ctx.traj_size
            max_epoch = ctx.max_epoch
            input_size = ctx.input_size
            output_size = ctx.output_size
            training_samples = ctx.training_samples
            alpha0 = ctx.alpha0

            current_epoch = AutoGradEndtoEnd._current_epoch

            _verbose = (AutoGradEndtoEnd._current_batch == 0)
            if _verbose:
                print("[BACKWARD AutoGrad] CNN features shape: ", features_flat.shape)
                print("y_label shape: ", y_labels.shape)

            # Reuse formatted arrays cached from forward pass
            train_x = ctx.train_x
            train_y = ctx.train_y

            # Call RayBNN backward pass
            try:
                if _verbose:
                    print(f"[BACKWARD AutoGrad] Calling RayBNN backward using reshaped CNN features shape: {train_x.shape}")

                ## GET dL/dXCNN
                # DEFERRED UPDATE: Rust backward now returns:
                #   (dL_dX, raw_negated_bnn_grad, current_params, dummy)
                # BNN weights are NOT updated inside Rust anymore.
                # We store the gradient and apply the update AFTER CNN optimizer.step().
                grad_result, raw_bnn_grad, current_params, _ = raybnn_python.state_space_backward_group2(
                    train_x, train_y, traj_size, max_epoch, alpha0, AutoGradEndtoEnd._arch_search, current_epoch
                )
                # Store raw BNN gradient for deferred update (applied after CNN optimizer.step)
                AutoGradEndtoEnd._pending_bnn_grad = np.array(raw_bnn_grad, dtype=np.float32).flatten()
                AutoGradEndtoEnd._pending_bnn_params = np.array(current_params, dtype=np.float32).flatten()
                if _verbose:
                    print("grad_result shape: ", grad_result.shape)
                    print("grad_output shape: ", grad_output.shape)
                # Convert grad_result to numpy array if needed (BEFORE using it)
                if not isinstance(grad_result, np.ndarray):
                    grad_result = np.array(grad_result, dtype=np.float32)

                grad_result_reshaped = grad_result[:, :, 0, 0].T

                assert not np.isnan(grad_result).any(), "NaN in gradients!"

                # Convert dL/dXCNN from numpy arrays to Pytorch Tensors
                grad_features = torch.from_numpy(grad_result_reshaped).to(features_flat.device)

                assert grad_features.shape == features_flat.shape, \
                f"Gradient shape {grad_features.shape} doesn't match features {features_flat.shape}"

                # RayBNN gradient diagnostic (first batch only)
                if _verbose:
                    raybnn_grad_magnitude = grad_features.abs().mean().item()
                    grad_per_sample_norm = grad_features.norm(dim=1)
                    print("\n" + "="*70)
                    print(f"DIAGNOSTIC: dL/dX_raybnn (Epoch {current_epoch+1}, Batch {AutoGradEndtoEnd._current_batch})")
                    print(f"  Shape: {grad_features.shape}  Mean: {grad_features.mean().item():.8f}  Std: {grad_features.std().item():.8f}")
                    print(f"  Min: {grad_features.min().item():.8f}  Max: {grad_features.max().item():.8f}")
                    print(f"  RayBNN grad magnitude: {raybnn_grad_magnitude:.8f}")
                    print(f"  Per-sample grad norm std: {grad_per_sample_norm.std().item():.8f}")
                    if grad_per_sample_norm.std().item() < 1e-8:
                        print("  CRITICAL: All samples get identical gradients!")
                    correlation = torch.corrcoef(
                        torch.stack([grad_features.flatten(), features_flat.flatten()])
                    )[0, 1].item()
                    print(f"  Gradient-feature correlation: {correlation:.6f}")
                    print("="*70)

            except Exception as e:
                print(f"[AUTOGRAD BACKWARD] Error calling RayBNN backward: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: return pass-through gradients
                grad_features = torch.zeros_like(features_flat)  # Create fallback gradient
                print("[AUTOGRAD BACKWARD] Using zero gradients as fallback")

            return grad_features, None, None, None, None, None, None, None, None

    class EndtoEndTrainer(nn.Module):
        def __init__(self, batch_size, traj_size, max_epoch, input_size, output_size, training_samples, alpha0):
            super().__init__()
            self.cnn = CNN_EEG()
            # Store RayBNN parameters for AutoGradEndtoEnd
            self.batch_size = batch_size
            self.traj_size = traj_size
            self.max_epoch = max_epoch
            self.input_size = input_size
            self.output_size = output_size
            self.training_samples = training_samples
            self.alpha0 = alpha0

        def update_epoch(self, epoch):
        #Update the current epoch for the autograd function
            AutoGradEndtoEnd._current_epoch = epoch

        def update_batch(self, batch):
        #Update the current batch for the autograd function
            AutoGradEndtoEnd._current_batch = batch

        def forward(self, raw_eeg, y_labels, verbose=True):
        # Move inputs to the same device as CNN
            raw_eeg = raw_eeg.to(self.device)
            y_labels = y_labels.to(self.device)
        # CNN forward pass with mixed precision (float16 on GPU for Conv2d speedup)
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                features = self.cnn(raw_eeg, y_labels, verbose)
            features = features.float()  # Ensure float32 at RayBNN boundary

            # Print CNN Output (Features) Before RayBNN ===
            if verbose:
                print("\n" + "="*70)
                print("DIAGNOSTIC: CNN Output (Features) BEFORE entering RayBNN")
                print("="*70)
                print(f"  Shape: {features.shape}")
                print(f"  Mean: {features.mean().item():.6f}")
                print(f"  Std: {features.std().item():.6f}")
                print(f"  Min: {features.min().item():.6f}")
                print(f"  Max: {features.max().item():.6f}")
                # Check if features are discriminative (different inputs -> different features)
                feature_variance_per_sample = features.var(dim=0).mean().item()
                feature_variance_per_feature = features.var(dim=1).mean().item()
                print(f"  Variance across samples (per feature): {feature_variance_per_sample:.6f}")
                print(f"  Variance across features (per sample): {feature_variance_per_feature:.6f}")
                if feature_variance_per_sample < 1e-6:
                    print("  CRITICAL: All samples produce nearly identical features!")
                else:
                    print("  Features vary across samples")
                print("="*70 + "\n")
            output = AutoGradEndtoEnd.apply(
                features,          # CNN features (batch, 256)
                y_labels,          # labels
                self.batch_size,   # batch size
                self.traj_size,    # trajectory size
                self.max_epoch,    # max epochs
                self.input_size,   # input size
                self.output_size,  # output size
                self.training_samples,
                self.alpha0
            )
            return output

    # Store arch_search on AutoGradEndtoEnd so it persists across forward/backward calls
    AutoGradEndtoEnd._arch_search = arch_search

    end_to_end_model = EndtoEndTrainer(batch_size, traj_size, max_epoch, input_size, output_size, training_samples, alpha0)
    end_to_end_model._autograd_cls = AutoGradEndtoEnd  # Store reference for deferred BNN update
    end_to_end_model.device = device
    end_to_end_model.to(device)

    return end_to_end_model, train_dataset, test_dataset, alpha0, batch_size, device


def apply_deferred_bnn_update(alpha0, autograd_cls):
    """Apply BNN weight update AFTER CNN optimizer.step().

    This fixes the 'moving target' problem: both CNN and BNN weight updates
    are now based on the same forward pass state, rather than BNN being
    updated inside backward() before CNN gets its gradients applied.

    Uses CPU Adam (matching gd_f32.rs) with persistent mt/vt state.

    Args:
        alpha0: BNN learning rate
        autograd_cls: The AutoGradEndtoEnd class (defined inside main())
    """
    bnn_grad = autograd_cls._pending_bnn_grad
    bnn_params = autograd_cls._pending_bnn_params

    if bnn_grad is None or bnn_params is None:
        return  # No pending update

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    grad_count = len(bnn_grad)
    param_count = len(bnn_params)

    # Initialize or load Adam state
    if autograd_cls._adam_mt is None or len(autograd_cls._adam_mt) != grad_count:
        autograd_cls._adam_mt = np.zeros(grad_count, dtype=np.float32)
        autograd_cls._adam_vt = np.zeros(grad_count, dtype=np.float32)

    mt = autograd_cls._adam_mt
    vt = autograd_cls._adam_vt

    # Apply Adam on CPU (matching gd_f32.rs adam())
    # Vectorized Adam (bnn_grad already negated from Rust: grad = -1.0 * grad)
    update_len = min(param_count, grad_count)
    g = bnn_grad[:update_len]
    mt[:update_len] = beta1 * mt[:update_len] + (1.0 - beta1) * g
    vt[:update_len] = beta2 * vt[:update_len] + (1.0 - beta2) * g * g
    nmt = mt[:update_len] / (1.0 - beta1)
    nvt = np.sqrt(vt[:update_len] / (1.0 - beta2)) + epsilon
    bnn_params[:update_len] += alpha0 * (nmt / nvt)  # grad already negated, so ADD

    # Persist updated params back to arch_search
    # depythonize on the Rust side requires a Python Sequence (list), not a numpy ndarray.
    autograd_cls._arch_search["neural_network"]["network_params"]["data"] = bnn_params.tolist()

    # Persist Adam state
    autograd_cls._adam_mt = mt
    autograd_cls._adam_vt = vt
    autograd_cls._arch_search["adam_mt"] = mt.tolist()
    autograd_cls._arch_search["adam_vt"] = vt.tolist()

    # Clear pending grad
    autograd_cls._pending_bnn_grad = None
    autograd_cls._pending_bnn_params = None


def train_ete_model(model, train_dataset, test_dataset, alpha0, batch_size, max_epoch, mode="both", eval_every=2):

    device = model.device

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,  # drop_last ensures all batches are exactly batch_size
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
        persistent_workers=True
    )

    # ========== AREA 1 Print CNN Parameter Updates ==========
    print("\n" + "="*70)
    print("AREA 1 DIAGNOSTIC: Checking CNN Parameter Updates")
    print("="*70)

    # Store initial CNN parameters
    cnn_params_init = {
        name: param.clone().detach()
        for name, param in model.cnn.named_parameters()
    }

    print("\n=== Initial CNN Parameters ===")
    for name, param in cnn_params_init.items():
        print(f"{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {param.mean().item():.6f}")
        print(f"  Std: {param.std().item():.6f}")
        print(f"  Min/Max: [{param.min().item():.6f}, {param.max().item():.6f}]")

    ## Mode Parameter
    # mode: "both" - Train CNN + RayBNN
    # mode: "cnn_only" - Train CNN, freeze RayBNN (alpha0=0)
    # mode: "raybnn_only" - Freeze CNN, train RayBNN only
    # mode: "frozen" - Freeze both (sanity check)
    if mode == "cnn_only":
        for param in model.cnn.parameters():
            param.requires_grad = True
        lr = 0.0008
        alpha0 = 0.0 # Freeze RayBNN
        print(f"MODE: Train CNN only with CNN_lr={lr}, RayBNN_lr={alpha0}")

    elif mode =="raybnn_only":
        for param in model.cnn.parameters():
            param.requires_grad = False
        lr = 0.0
        print(f"MODE: Train RayBNN only with lr={alpha0}, CNN_lr={lr})")

    elif mode == "frozen":
        for param in model.cnn.parameters():
            param.requires_grad = False
        lr = 0.0
        alpha0 = 0.0
        print(f"MODE: Both frozen (lr={lr}, alpha0={alpha0})")

    else:
        for param in model.cnn.parameters():
            param.requires_grad = True
        lr = 0.0008
        # alpha0 stays as passed in
        print(f"MODE: Train both (lr={lr}, alpha0={alpha0})")

    # Propagate alpha0 to model so backward() uses the mode-adjusted value
    model.alpha0 = alpha0

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=lr
    ) if lr > 0 else None
    criterion = torch.nn.CrossEntropyLoss()

    # Set model to training mode
    model.train()

    # ==================== MODEL COMPLEXITY SUMMARY ====================
    process = psutil.Process()
    cnn_total_params = sum(p.numel() for p in model.cnn.parameters())
    cnn_trainable_params = sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)
    bnn_param_count = len(model._autograd_cls._arch_search["neural_network"]["network_params"]["data"])
    total_params = cnn_total_params + bnn_param_count

    print(f"\n{'='*70}")
    print(f" MODEL COMPLEXITY SUMMARY")
    print(f"{'='*70}")
    print(f"  CNN parameters      : {cnn_total_params:>10,}  (trainable: {cnn_trainable_params:,})")
    print(f"  RayBNN parameters   : {bnn_param_count:>10,}")
    print(f"  Total parameters    : {total_params:>10,}")
    print(f"  CNN layers:")
    for name, param in model.cnn.named_parameters():
        print(f"    {name:20s}  shape={str(list(param.shape)):20s}  params={param.numel():,}")
    mem_before = process.memory_info().rss / 1024**2
    print(f"  Memory before training: {mem_before:.1f} MB")
    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"{'='*70}")

    print(f"\n{'='*50}")
    print(f" TRAINING: {max_epoch} epochs, batch_size={batch_size}")
    print(f" alpha0={alpha0}, lr={lr}")
    print(f"{'='*50}")

    batch_idx = len(train_loader)
    training_start_time = time.perf_counter()

    # Accumulators for final summary
    all_epoch_times = []
    all_epoch_losses = []
    all_epoch_accs = []
    all_test_losses = []
    all_test_accs = []
    all_test_kappas = []

    for epoch in range(max_epoch):

        model.update_epoch(epoch)
        epoch_start_time = time.perf_counter()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Per-epoch component timing accumulators
        epoch_t_fwd = 0.0      # CNN forward + RayBNN forward
        epoch_t_bwd = 0.0      # backward (RayBNN backward + CNN backward)
        epoch_t_opt = 0.0      # optimizer.step() + deferred BNN update
        epoch_t_other = 0.0    # diagnostics, logging, etc.

        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            if optimizer:
                optimizer.zero_grad()

            # Track current batch for backward diagnostics
            model.update_batch(i)
            # Reduce verbosity - only verbose for first batch of first epoch
            verbose = (i == 0 and epoch == 0)

            # ---- TIMING: Forward pass (CNN + RayBNN) ----
            t_fwd_start = time.perf_counter()
            output = model(batch_x, batch_y, verbose=verbose)
            t_fwd_end = time.perf_counter()
            epoch_t_fwd += (t_fwd_end - t_fwd_start)

            # ========== AREA 4: RayBNN Output (Yhat) Analysis after CNN->RayBNN forward ==========
            if i == 0 and epoch == 0:  # First batch of first epoch only
                print("\n" + "="*70)
                print(f"AREA 4 DIAGNOSTIC: RayBNN Yhat (CNN->RayBNN->Softmax input) (Epoch {epoch+1}, Batch {i})")
                print("="*70)

                with torch.no_grad():
                    print(f"\nOutput Statistics:")
                    print(f"  Shape: {output.shape}")
                    print(f"  Range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                    print(f"  Mean: {output.mean().item():.4f}")
                    print(f"  Std: {output.std().item():.4f}")

                    # Prediction analysis
                    probs = torch.softmax(output, dim=1)
                    preds = output.argmax(dim=1)

                    # Entropy (max entropy for 4 classes = ln(4) = 1.386)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
                    print(f"  Entropy: {entropy:.4f}/1.386 (random=1.386)")

                    if entropy > 1.3:
                        print(f"  WARNING: High entropy - predictions are near-random!")
                    elif entropy < 0.7:
                        print(f"  EXCELLENT: Low entropy - confident predictions")
                    else:
                        print(f"  GOOD: Moderate entropy - learning in progress")

                print("="*70 + "\n")

            loss = criterion(output, batch_y)

            # CRITICAL FIX: Loss explosion detection
            if loss.item() > 10.0 or torch.isnan(loss) or torch.isinf(loss):
                print(f"LOSS EXPLOSION DETECTED: {loss.item():.3f}")
                print("Stopping training to prevent further damage")
                return model  # Early termination

            # ---- TIMING: Backward pass ----
            t_bwd_start = time.perf_counter()
            if mode in ("both", "cnn_only"):
                loss.backward()
            t_bwd_end = time.perf_counter()
            epoch_t_bwd += (t_bwd_end - t_bwd_start)

            # ========== AREA 2: Print Gradient Analysis ==========
            if i == 0 and epoch == 0:  # First batch of first epoch only
                print("\n" + "="*70)
                print(f"AREA 2 DIAGNOSTIC: CNN Parameters Gradients AFTER backprop (Epoch {epoch}, Batch {i})")
                print("="*70)

                print("\n=== CNN Gradients ===")
                for name, param in model.cnn.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        grad_max = param.grad.abs().max().item()
                        grad_min = param.grad.abs().min().item()

                        print(f"\n{name}:")
                        print(f"  Gradient mean: {grad_mean:.8f}")
                        print(f"  Gradient max: {grad_max:.8f}")
                        print(f"  Gradient min: {grad_min:.8f}")

                        # Diagnose gradient issues
                        if grad_mean < 1e-6:
                            print(f"  CRITICAL: Vanishing gradients! (mean < 1e-6)")
                        elif grad_mean > 1.0:
                            print(f"  WARNING: Large gradients (mean > 1.0)")
                        else:
                            print(f"  Healthy gradient magnitude")

                        # Check for all-zero gradients
                        if (param.grad == 0).all():
                            print(f"  CRITICAL: ALL gradients are zero!")
                    else:
                        print(f"\n{name}: NO GRADIENT (requires_grad might be False)")

                print("\n" + "="*70 + "\n")

            # ---- TIMING: Optimizer step ----
            t_opt_start = time.perf_counter()
            if optimizer:
                optimizer.step()

            # DEFERRED BNN UPDATE: Apply BNN weight update AFTER CNN optimizer.step()
            # This ensures both optimizers compute gradients against the same model state.
            if mode in ("both", "raybnn_only"):
                apply_deferred_bnn_update(alpha0, model._autograd_cls)
            t_opt_end = time.perf_counter()
            epoch_t_opt += (t_opt_end - t_opt_start)

            # === CNN Param Update Check (first batch of first epoch) ===
            if i == 0 and epoch == 0:
                print(f"\n=== CNN Parameter Delta (Epoch {epoch+1}, after optimizer.step) ===")
                for name, param in model.cnn.named_parameters():
                    diff = (param.detach() - cnn_params_init[name]).abs()
                    print(f"  {name}: mean_delta={diff.mean().item():.8f}, max_delta={diff.max().item():.8f}")

            # Accumulate metrics (no_grad avoids unnecessary graph tracking)
            epoch_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(output.data, 1)
                epoch_total += batch_y.size(0)
                epoch_correct += (predicted == batch_y).sum().item()

        # ===== END OF EPOCH SUMMARY =====
        epoch_time = time.perf_counter() - epoch_start_time
        epoch_t_other = epoch_time - (epoch_t_fwd + epoch_t_bwd + epoch_t_opt)
        final_epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        avg_epoch_loss = epoch_loss / max(batch_idx, 1)
        samples_per_sec = epoch_total / epoch_time if epoch_time > 0 else 0

        # Store for final summary
        all_epoch_times.append(epoch_time)
        all_epoch_losses.append(avg_epoch_loss)
        all_epoch_accs.append(final_epoch_acc)

        # ===== PER-EPOCH TEST EVALUATION (every eval_every epochs + last epoch) =====
        run_eval = (epoch % eval_every == 0) or (epoch == max_epoch - 1)
        test_eval_time = 0.0

        if run_eval:
            test_eval_start = time.perf_counter()
            model.eval()
            test_loss_sum = 0.0
            test_correct = 0
            test_total = 0
            test_num_batches = 0
            all_preds_epoch = []
            all_labels_epoch = []
            # Pre-allocate padding tensors once (reused across batches)
            _pad_x = None
            _pad_y = None
            with torch.no_grad():
                for ti, (tbx, tby) in enumerate(test_loader):
                    tbx = tbx.to(device, non_blocking=True)
                    tby = tby.to(device, non_blocking=True)
                    actual_batch = tbx.size(0)
                    # Pad to batch_size if needed (RayBNN requires fixed batch size)
                    if actual_batch < batch_size:
                        if _pad_x is None:
                            _pad_x = torch.zeros(batch_size, *tbx.shape[1:], device=device)
                            _pad_y = torch.zeros(batch_size, dtype=tby.dtype, device=device)
                        _pad_x[:actual_batch] = tbx
                        _pad_y[:actual_batch] = tby
                        pad_x, pad_y = _pad_x, _pad_y
                        model.update_batch(ti)
                        tout = model(pad_x, pad_y, verbose=False)
                        tout = tout[:actual_batch]
                    else:
                        model.update_batch(ti)
                        tout = model(tbx, tby, verbose=False)
                    test_loss_sum += criterion(tout, tby).item()
                    test_correct += (tout.argmax(dim=1) == tby).sum().item()
                    test_total += actual_batch
                    test_num_batches += 1
                    all_preds_epoch.append(tout.argmax(dim=1).cpu().numpy())
                    all_labels_epoch.append(tby.cpu().numpy())

            all_preds_epoch = np.concatenate(all_preds_epoch)
            all_labels_epoch = np.concatenate(all_labels_epoch)
            test_kappa = cohen_kappa_score(all_labels_epoch, all_preds_epoch)

            avg_test_loss = test_loss_sum / max(test_num_batches, 1)
            test_acc = test_correct / test_total if test_total > 0 else 0.0
            model.train()
            test_eval_time = time.perf_counter() - test_eval_start
        else:
            # Carry forward last known test metrics
            avg_test_loss = all_test_losses[-1] if all_test_losses else 0.0
            test_acc = all_test_accs[-1] if all_test_accs else 0.0
            test_kappa = all_test_kappas[-1] if all_test_kappas else 0.0

        all_test_losses.append(avg_test_loss)
        all_test_accs.append(test_acc)
        all_test_kappas.append(test_kappa)

        eval_tag = "" if run_eval else " (cached)"
        print(f"\nEpoch {epoch+1}/{max_epoch} - loss: {avg_epoch_loss:.4f} - acc: {final_epoch_acc:.4f} - "
              f"test_loss: {avg_test_loss:.4f}{eval_tag} - test_acc: {test_acc:.4f}{eval_tag} - kappa: {test_kappa:.4f}{eval_tag} - "
              f"time: {epoch_time:.2f}s ({samples_per_sec:.0f} samp/s) | "
              f"fwd: {epoch_t_fwd:.2f}s  bwd: {epoch_t_bwd:.2f}s  opt: {epoch_t_opt:.2f}s  other: {epoch_t_other:.2f}s  eval: {test_eval_time:.2f}s")

    # ==================== FINAL TRAINING SUMMARY ====================
    total_training_time = time.perf_counter() - training_start_time
    mem_after = process.memory_info().rss / 1024**2

    print(f"\n{'='*70}")
    print(f" FINAL TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  Total training time  : {total_training_time:.2f}s ({total_training_time/60:.1f} min)")
    print(f"  Epochs completed     : {max_epoch}")
    print(f"  Avg time per epoch   : {sum(all_epoch_times)/len(all_epoch_times):.2f}s")
    print(f"  Avg throughput       : {max_epoch * epoch_total / total_training_time:.0f} samples/sec")
    print(f"  Final accuracy       : {all_epoch_accs[-1]:.4f}")
    print(f"  Final loss           : {all_epoch_losses[-1]:.4f}")
    print(f"  Best accuracy        : {max(all_epoch_accs):.4f} (epoch {all_epoch_accs.index(max(all_epoch_accs))+1})")
    print(f"  Best loss            : {min(all_epoch_losses):.4f} (epoch {all_epoch_losses.index(min(all_epoch_losses))+1})")
    print(f"  Final test loss      : {all_test_losses[-1]:.4f}")
    print(f"  Final test accuracy  : {all_test_accs[-1]:.4f}")
    print(f"  Final test kappa     : {all_test_kappas[-1]:.4f}")
    print(f"  Best test accuracy   : {max(all_test_accs):.4f} (epoch {all_test_accs.index(max(all_test_accs))+1})")
    print(f"  Best test loss       : {min(all_test_losses):.4f} (epoch {all_test_losses.index(min(all_test_losses))+1})")
    print(f"  Best test kappa      : {max(all_test_kappas):.4f} (epoch {all_test_kappas.index(max(all_test_kappas))+1})")
    print(f"  Model complexity     : {total_params:,} total params (CNN: {cnn_total_params:,}, RayBNN: {bnn_param_count:,})")
    print(f"  Memory: before={mem_before:.1f} MB, after={mem_after:.1f} MB, delta={mem_after-mem_before:.1f} MB")
    print(f"{'='*70}")

    final_train_acc = all_epoch_accs[-1] if all_epoch_accs else 0.0

    return model, {
        "epoch_losses": all_epoch_losses,
        "epoch_accs": all_epoch_accs,
        "test_losses": all_test_losses,
        "test_accs": all_test_accs,
        "test_kappas": all_test_kappas,
        "training_time_sec": total_training_time,
        "final_train_acc": final_train_acc,
    }

def evaluate_model(model, test_dataset, batch_size=1000):

    ## Return dict with accuracy, loss, kappa, etc
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
        persistent_workers=True
    )

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    print(f"\n{'='*70}")
    print(f" EVALUATION ON TEST SET ({len(test_dataset):,} samples)")
    print(f"{'='*70}")

    inference_start = time.perf_counter()
    _pad_x = None
    _pad_y = None
    device = model.device
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            actual_batch = batch_x.size(0)
            # Pad to batch_size if needed (RayBNN requires fixed batch size)
            if actual_batch < batch_size:
                if _pad_x is None:
                    _pad_x = torch.zeros(batch_size, *batch_x.shape[1:], device=device)
                    _pad_y = torch.zeros(batch_size, dtype=batch_y.dtype, device=device)
                _pad_x[:actual_batch] = batch_x
                _pad_y[:actual_batch] = batch_y
                pad_x, pad_y = _pad_x, _pad_y
                model.update_batch(i)
                output = model(pad_x, pad_y, verbose=False)
                output = output[:actual_batch]
            else:
                model.update_batch(i)
                output = model(batch_x, batch_y, verbose=False)

            loss = criterion(output, batch_y)
            total_loss += loss.item()
            num_batches += 1

            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    inference_time = time.perf_counter() - inference_start

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / max(num_batches, 1)
    throughput = len(all_labels) / inference_time

    # Cohen's Kappa
    kappa_overall = cohen_kappa_score(all_labels, all_preds)
    kappa_per_class = kappa_metric(all_labels, all_preds, n_cl=4)

    # Per-class metrics
    class_names = ["Wake(0)", "N1(1)", "N2(2)", "Microsleep(3)"]
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    # Print results
    print(f"\n{'='*70}")
    print(f" TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy        : {accuracy:.4f} ({int(accuracy * len(all_labels))}/{len(all_labels)})")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"  F1-Score        : {f1:.4f}")
    print(f"  Average loss    : {avg_loss:.4f}")
    print(f"  Cohen Kappa     : {kappa_overall:.4f}")
    print(f"  Kappa per class : {kappa_per_class}")
    print(f"  Inference time  : {inference_time:.2f}s")
    print(f"  Throughput      : {throughput:.0f} samples/sec")

    print(f"\n  Classification Report (per-class precision/recall/F1):")
    print(report)
    print(f"{'='*70}")

    # Set back to training mode in case user continues training
    model.train()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_loss": avg_loss,
        "kappa_overall": kappa_overall,
        "kappa_per_class": kappa_per_class.tolist(),
        "inference_time": inference_time,
        "throughput": throughput,
    }

def plot_losses(history, save_path="mwt_cnn_raybnn_train_test_loss.png"):
    """Plot train and test loss per epoch."""
    train_losses = history['epoch_losses']
    test_losses = history['test_losses']
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r--', label='Test Loss')
    ax1.annotate(f'{train_losses[-1]:.4f}', xy=(epochs[-1], train_losses[-1]),
                 xytext=(5, 5), textcoords='offset points', color='blue', fontsize=9)
    ax1.annotate(f'{test_losses[-1]:.4f}', xy=(epochs[-1], test_losses[-1]),
                 xytext=(5, -15), textcoords='offset points', color='red', fontsize=9)
    ax1.set_title('CNN+RayBNN on MWT EEG: Train vs Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend()
    ax1.grid(True)

    # Kappa plot
    if 'test_kappas' in history:
        test_kappas = history['test_kappas']
        ax2.plot(epochs, test_kappas, 'g-', label='Test Cohen Kappa')
        ax2.annotate(f'{test_kappas[-1]:.4f}', xy=(epochs[-1], test_kappas[-1]),
                     xytext=(5, 5), textcoords='offset points', color='green', fontsize=9)
        ax2.set_title('CNN+RayBNN on MWT EEG: Test Cohen Kappa')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Cohen Kappa')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend()
        ax2.grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Loss plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':

    model_run_start = time.perf_counter()
    end_to_end_model, train_dataset, test_dataset, alpha0, batch_size, device = main()
    trained_model, train_history = train_ete_model(end_to_end_model, train_dataset,
    test_dataset, alpha0, batch_size=batch_size,
    max_epoch=50, mode="both")

    results = evaluate_model(trained_model, test_dataset, batch_size=batch_size)

    training_time_sec = train_history.get('training_time_sec', 0.0)
    testing_time_sec = results.get('inference_time', 0.0)
    model_train_test_total_sec = training_time_sec + testing_time_sec

    # Plot train and test loss + kappa
    plot_losses(train_history, save_path="mwt_cnn_raybnn_train_test_loss.png")

    end_to_end_runtime_sec = time.perf_counter() - model_run_start

    print(f"\nFinal Train Accuracy: {train_history['final_train_acc']:.4f}")
    print(f"Final Test Accuracy : {results['accuracy']:.4f}")
    print(f"Final Test Kappa    : {results['kappa_overall']:.4f}")
    print(f"Total Train+Test Runtime: {model_train_test_total_sec:.2f} sec ({model_train_test_total_sec/60:.2f} min)")
    print("Done without errors!")

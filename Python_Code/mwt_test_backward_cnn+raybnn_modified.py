import raybnn_python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import os
import math
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

    def __init__(self, mat_dir, file_list, w_len=1600, stride=1, traj_size=1):
        """
        Args:
            mat_dir: path to directory containing .mat files
            file_list: list of .mat filenames to include
            w_len: half-window size in samples (1600 = 8s at 200Hz, full window = 16s)
            stride: take every Nth sample (stride=16 reduces 28.8M to ~1.8M with negligible info loss)
            traj_size: number of temporal sub-windows per sample (1=original, 4=split 16s into 4x4s)
        """
        self.w_len = w_len
        self.data_dim = w_len * 2  # 3200 samples = 16 seconds
        self.stride = stride
        self.traj_size = traj_size
        self.sub_win_len = self.data_dim // traj_size  # 3200//4 = 800 when traj_size=4

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

        label = self.labels[subj_idx][sample_idx]

        if self.traj_size > 1:
            # Split 16s window into traj_size sub-windows (e.g., 4x4s = 4x800 samples)
            # Output shape: (traj_size, 2, sub_win_len, 1)
            O1_subs = window_O1.reshape(self.traj_size, self.sub_win_len)  # (4, 800)
            O2_subs = window_O2.reshape(self.traj_size, self.sub_win_len)  # (4, 800)
            window = np.stack([O1_subs, O2_subs], axis=1)  # (4, 2, 800)
            window = window[:, :, :, np.newaxis]  # (4, 2, 800, 1)
        else:
            # Original: (2, 3200, 1)
            window = np.stack([window_O1, window_O2], axis=0)  # (2, 3200)
            window = window[:, :, np.newaxis]  # (2, 3200, 1)

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


def load_partition_files(partition_dir, part_num):
    """Load train/val/test file lists from Xuan Chen's partition .mat file.

    Args:
        partition_dir: directory containing file_sets_part{1-4}.mat
        part_num: partition number (1-4)
    Returns:
        (train_files, val_files, test_files) as lists of .mat filenames
    """
    mat = spio.loadmat(os.path.join(partition_dir, f'file_sets_part{part_num}.mat'))
    files_train = []
    for i in range(len(mat['files_train'])):
        file = [str(''.join(l)) for la in mat['files_train'][i] for l in la]
        files_train.extend(file)
    files_val = []
    for i in range(len(mat['files_val'])):
        file = [str(''.join(l)) for la in mat['files_val'][i] for l in la]
        files_val.extend(file)
    files_test = []
    for i in range(len(mat['files_test'])):
        file = [str(''.join(l)) for la in mat['files_test'][i] for l in la]
        files_test.extend(file)
    print(f"Fold {part_num}: train={len(files_train)}, val={len(files_val)}, test={len(files_test)} subjects")
    return files_train, files_val, files_test


# ==================== CNN for EEG (simplified, mirrors MNIST CNN structure) ====================

class CNN_EEG(nn.Module):
    """Deep Conv2D CNN for EEG signals (12 conv layers, ported from Xuan Chen's Keras model).

    Input:  (batch, 2, H, 1) -- 2 EEG channels, H time steps (3200 or 800), width=1
    Output: (batch, 256) -- feature vector

    Architecture (from myModel.py build_model/cnn_block):
        GaussianNoise(0.0005) -- training only
        Conv2d(2->32,  (3,1)) + BN + ReLU + MaxPool(2,1)
        Conv2d(32->64, (3,1)) + BN + ReLU + MaxPool(2,1)
        4x Conv2d(64->128,  (3,1)) + BN + ReLU + MaxPool(2,1)
        6x Conv2d(128->256, (3,1)) + BN + ReLU + MaxPool(2,1)
        AdaptiveAvgPool2d((1,1)) + Flatten
        Dense(256) + BN + Dropout(0.5)
    """

    def __init__(self, feature_dim=256):
        super(CNN_EEG, self).__init__()
        self.noise_std = 0.0005

        # Block 1: 2 -> 32 -> 64
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-3)

        # Block 2: 4x (64 -> 128)
        self.conv3_layers = nn.ModuleList()
        self.bn3_layers = nn.ModuleList()
        in_ch = 64
        for i in range(4):
            self.conv3_layers.append(nn.Conv2d(in_ch, 128, kernel_size=(3, 1), padding=(1, 0)))
            self.bn3_layers.append(nn.BatchNorm2d(128, momentum=0.01, eps=1e-3))
            in_ch = 128

        # Block 3: 6x (128 -> 256)
        self.conv4_layers = nn.ModuleList()
        self.bn4_layers = nn.ModuleList()
        in_ch = 128
        for i in range(6):
            self.conv4_layers.append(nn.Conv2d(in_ch, 256, kernel_size=(3, 1), padding=(1, 0)))
            self.bn4_layers.append(nn.BatchNorm2d(256, momentum=0.01, eps=1e-3))
            in_ch = 256

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Head: Flatten -> Dense(256) + BN + Dropout
        self.fc1 = nn.Linear(256, feature_dim)
        self.bn_fc = nn.BatchNorm1d(feature_dim, momentum=0.01, eps=1e-3)
        self.drop = nn.Dropout(0.5)

        # Xavier normal init (equiv. to Keras glorot_normal)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y_labels=None, verbose=False):
        """
        Args:
            x: (batch, 2, H, 1) EEG windows (H=3200 or H=800)
            y_labels: not used, kept for API compatibility
            verbose: print shape diagnostics
        """
        # Gaussian noise (training only)
        if self.training:
            x = x + self.noise_std * torch.randn_like(x)

        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        if verbose:
            print(f"After conv1+BN+ReLU+pool: {x.shape}")
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        if verbose:
            print(f"After conv2+BN+ReLU+pool: {x.shape}")

        # Block 2: 4x conv(128)
        for i in range(4):
            x = self.pool(F.relu(self.bn3_layers[i](self.conv3_layers[i](x))))
        if verbose:
            print(f"After 4x conv3+BN+ReLU+pool: {x.shape}")

        # Block 3: 6x conv(256)
        for i in range(6):
            x = self.pool(F.relu(self.bn4_layers[i](self.conv4_layers[i](x))))
        if verbose:
            print(f"After 6x conv4+BN+ReLU+pool: {x.shape}")

        # Adaptive pool + flatten + FC head
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)  # (batch, 256)
        x = F.relu(self.fc1(x))
        x = self.bn_fc(x)
        features_flat = self.drop(x)

        if verbose:
            print(f"CNN output features: {features_flat.shape}")  # (batch, 256)

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

def main(fold_num=None, traj_size=4, proc_num=4, batch_size=1000):
    """Setup model and data for one fold.

    Args:
        fold_num: 1-4 to use Xuan Chen's partition scheme, None for 60/16 random split
        traj_size: number of temporal sub-windows (4 = split 16s into 4x4s)
        proc_num: RayBNN processing steps per trajectory step
        batch_size: samples per batch
    """

    ## Parameter setting for MWT EEG dataset
    dir_path = "/tmp/"
    max_input_size = 256
    input_size = 256

    max_output_size = 4
    output_size = 4

    max_neuron_size = 2000

    active_size = 1000

    training_samples = 1  # Each AutoGrad call processes 1 batch (batch_size samples)
    crossval_samples = 1
    testing_samples = 1

    alpha0 = 0.0005

    # IMPORTANT: Create RayBNN architecture BEFORE PyTorch initializes its CUDA context.
    # ArrayFire (cuSPARSE) must initialize first to avoid NULL handle segfault.
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

    # Now safe to initialize PyTorch CUDA context (ArrayFire already owns the device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ## Load MWT EEG dataset
    mat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mwt_eeg')
    mat_dir = os.path.normpath(mat_dir)
    print(f"MWT EEG data directory: {mat_dir}")

    if fold_num is not None:
        partition_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..', 'xuanchen_code', 'c', 'CNN+RayBNN')
        partition_dir = os.path.normpath(partition_dir)
        train_files, val_files, test_files = load_partition_files(partition_dir, fold_num)
        print("Loading training dataset...")
        train_dataset = MWTDataset(mat_dir, train_files, stride=16, traj_size=traj_size)
        print("Loading validation dataset...")
        val_dataset = MWTDataset(mat_dir, val_files, stride=16, traj_size=traj_size)
        print("Loading test dataset...")
        test_dataset = MWTDataset(mat_dir, test_files, stride=16, traj_size=traj_size)
    else:
        train_files, test_files = split_subjects(mat_dir, n_train=60, seed=42)
        print("Loading training dataset...")
        train_dataset = MWTDataset(mat_dir, train_files, stride=16, traj_size=traj_size)
        val_dataset = None
        print("Loading test dataset...")
        test_dataset = MWTDataset(mat_dir, test_files, stride=16, traj_size=traj_size)

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

            # features_flat shape:
            #   traj_size==1: (batch_size, 256)
            #   traj_size>1:  (batch_size, traj_size, 256)
            # y_labels shape: (batch_size,)

            # Convert X and Y to numpy arrays (Convert PyTorch -> NumPy)
            features_np = features_flat.detach().cpu().numpy()
            y_labels_np = y_labels.detach().cpu().numpy()

            # Create training arrays: (input_size, batch_size, traj_size, 1)
            train_x = np.zeros((input_size, batch_size, traj_size, 1), dtype=np.float32)
            train_y = np.zeros((output_size, batch_size, traj_size, 1), dtype=np.float32)

            indices = y_labels_np.astype(int)
            valid = (indices >= 0) & (indices < output_size)

            if traj_size > 1:
                # features_np: (batch_size, traj_size, 256)
                for t in range(traj_size):
                    train_x[:, :, t, 0] = features_np[:, t, :].T  # (256, batch_size)
                    train_y[indices[valid], np.where(valid)[0], t, 0] = 1.0
            else:
                # features_np: (batch_size, 256)
                train_x[:, :, 0, 0] = features_np.T  # (256, 1000)
                train_y[indices[valid], np.where(valid)[0], 0, 0] = 1.0

            _verbose = (AutoGradEndtoEnd._current_batch == 0)
            if _verbose:
                print("[FORWARD AutoGrad] CNN features reshaped as RayBNN input: ", train_x.shape)

            # Sync PyTorch CUDA before entering ArrayFire (prevents cuSPARSE context conflict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

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

            # Yhat shape from Rust: (output_size, batch_size, traj_size, 1)
            # Use last trajectory step for prediction (state-space model accumulates info)
            if traj_size > 1:
                # Take last traj step: Yhat[:, :, -1, 0] -> (output_size, batch_size) -> .T
                Yhat = Yhat_tensor[:, :, -1, 0].T  # (batch_size, output_size)
            else:
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
                # Sync PyTorch CUDA before entering ArrayFire backward
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

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

                # dL_dX from Rust: (input_size, batch_size, traj_steps, 1)
                # where traj_steps = traj_size + proc_num - 1
                # For traj_size>1: extract first traj_size slices (input gradients)
                # For traj_size==1: extract slice 0 only
                if traj_size > 1:
                    # (256, 1000, traj_size) -> transpose to (1000, traj_size, 256)
                    grad_result_reshaped = grad_result[:, :, :traj_size, 0].transpose(1, 2, 0)
                else:
                    grad_result_reshaped = grad_result[:, :, 0, 0].T

                assert not np.isnan(grad_result).any(), "NaN in gradients!"

                # Convert dL/dXCNN from numpy arrays to Pytorch Tensors
                grad_features = torch.from_numpy(grad_result_reshaped.copy()).to(features_flat.device)

                assert grad_features.shape == features_flat.shape, \
                f"Gradient shape {grad_features.shape} doesn't match features {features_flat.shape}"

                # RayBNN gradient diagnostic (first batch only)
                if _verbose:
                    raybnn_grad_magnitude = grad_features.abs().mean().item()
                    grad_flat = grad_features.reshape(batch_size, -1)
                    grad_per_sample_norm = grad_flat.norm(dim=1)
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

            if self.traj_size > 1:
                # raw_eeg: (batch, traj_size, 2, sub_win_len, 1)
                B, T = raw_eeg.shape[0], raw_eeg.shape[1]
                # Reshape to (batch*traj_size, 2, sub_win_len, 1) for CNN
                cnn_input = raw_eeg.reshape(B * T, *raw_eeg.shape[2:])
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    features_all = self.cnn(cnn_input, None, verbose)
                features_all = features_all.float()
                # Reshape back: (batch, traj_size, 256)
                features = features_all.reshape(B, T, -1)
            else:
                # raw_eeg: (batch, 2, 3200, 1)
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    features = self.cnn(raw_eeg, y_labels, verbose)
                features = features.float()

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
                feat_2d = features.reshape(features.size(0), -1)
                feature_variance_per_sample = feat_2d.var(dim=0).mean().item()
                feature_variance_per_feature = feat_2d.var(dim=1).mean().item()
                print(f"  Variance across samples (per feature): {feature_variance_per_sample:.6f}")
                print(f"  Variance across features (per sample): {feature_variance_per_feature:.6f}")
                if feature_variance_per_sample < 1e-6:
                    print("  CRITICAL: All samples produce nearly identical features!")
                else:
                    print("  Features vary across samples")
                print("="*70 + "\n")
            output = AutoGradEndtoEnd.apply(
                features,          # CNN features: (batch, 256) or (batch, traj_size, 256)
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

    return end_to_end_model, train_dataset, val_dataset, test_dataset, alpha0, batch_size, device


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
    # Try numpy array directly (depythonize supports Python sequence protocol).
    # If Rust panics here, revert to: bnn_params.tolist()
    autograd_cls._arch_search["neural_network"]["network_params"]["data"] = bnn_params

    # Persist Adam state
    autograd_cls._adam_mt = mt
    autograd_cls._adam_vt = vt
    autograd_cls._arch_search["adam_mt"] = mt
    autograd_cls._arch_search["adam_vt"] = vt

    # Clear pending grad
    autograd_cls._pending_bnn_grad = None
    autograd_cls._pending_bnn_params = None


def train_ete_model(model, train_dataset, test_dataset, alpha0, batch_size, max_epoch, mode="both", eval_every=2, val_dataset=None):

    device = model.device

    # Use val_dataset for in-training evaluation if available, else fall back to test_dataset
    eval_dataset = val_dataset if val_dataset is not None else test_dataset

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,  # drop_last ensures all batches are exactly batch_size
        persistent_workers=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
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
        lr = 0.001
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
        lr = 0.0005
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
                for ti, (tbx, tby) in enumerate(eval_loader):
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

    # ==================== Configuration ====================
    N_FOLDS = 4          # 4-fold subject-level CV (Xuan Chen's partition scheme)
    TRAJ_SIZE = 4        # 4 temporal sub-windows (4x4s = 16s)
    PROC_NUM = 4         # 4 processing steps per trajectory step
    BATCH_SIZE = 1000    # samples per batch
    MAX_EPOCH = 50       # training epochs per fold
    MODE = "both"        # train CNN + RayBNN end-to-end

    overall_start = time.perf_counter()
    fold_results = []
    fold_histories = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n{'='*80}")
        print(f" FOLD {fold}/{N_FOLDS}")
        print(f"{'='*80}")

        fold_start = time.perf_counter()

        # Setup model and data for this fold
        model, train_dataset, val_dataset, test_dataset, alpha0, batch_size, device = main(
            fold_num=fold, traj_size=TRAJ_SIZE, proc_num=PROC_NUM, batch_size=BATCH_SIZE
        )

        # Train end-to-end
        trained_model, train_history = train_ete_model(
            model, train_dataset, test_dataset, alpha0,
            batch_size=batch_size, max_epoch=MAX_EPOCH, mode=MODE,
            val_dataset=val_dataset
        )

        # Evaluate on test set (held-out subjects for this fold)
        results = evaluate_model(trained_model, test_dataset, batch_size=batch_size)

        fold_results.append(results)
        fold_histories.append(train_history)

        fold_time = time.perf_counter() - fold_start
        print(f"\nFold {fold} completed in {fold_time:.1f}s ({fold_time/60:.1f} min)")
        print(f"  Test Accuracy: {results['accuracy']:.4f}")
        print(f"  Test Kappa   : {results['kappa_overall']:.4f}")
        print(f"  Test F1      : {results['f1_score']:.4f}")

        # Free GPU memory before next fold
        del model, trained_model, train_dataset, val_dataset, test_dataset
        torch.cuda.empty_cache()

        # Plot per-fold losses
        plot_losses(train_history, save_path=f"mwt_cnn_raybnn_fold{fold}_loss.png")

    # ==================== Cross-Fold Summary ====================
    overall_time = time.perf_counter() - overall_start

    print(f"\n{'='*80}")
    print(f" {N_FOLDS}-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'kappa_overall', 'avg_loss']
    for m in metrics:
        values = [r[m] for r in fold_results]
        print(f"  {m:20s}: {np.mean(values):.4f} +/- {np.std(values):.4f}  (per fold: {[f'{v:.4f}' for v in values]})")

    # Per-class kappa
    all_kappas = np.array([r['kappa_per_class'] for r in fold_results])
    class_names = ["Wake(0)", "N1(1)", "N2(2)", "Microsleep(3)"]
    print(f"\n  Per-class kappa (mean +/- std):")
    for i, name in enumerate(class_names):
        vals = all_kappas[:, i]
        print(f"    {name:15s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    print(f"\n  Total runtime: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"  Config: traj_size={TRAJ_SIZE}, proc_num={PROC_NUM}, batch_size={BATCH_SIZE}, epochs={MAX_EPOCH}")
    print(f"{'='*80}")
    print("Done without errors!")

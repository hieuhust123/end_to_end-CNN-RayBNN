# Function Analysis: `reshape_cnn_features_to_raybnn_format`

## Current Function Signature (PROBLEMATIC)
```python
def reshape_cnn_features_to_raybnn_format(train_x, train_y, x_train, y_train, batch_size):
```

## Issues with Current Design:

1. **Pre-allocated arrays as parameters** (`train_x`, `train_y`):
   - Function shouldn't need pre-allocated arrays
   - Should create them internally based on dimensions
   - Makes function less reusable

2. **Missing required parameters**:
   - `input_size` - needed to create `train_x` shape
   - `output_size` - needed to create `train_y` shape  
   - `traj_size` - currently hardcoded as `0` in indexing
   - `num_samples` or `training_samples` - needed for array dimensions

3. **Parameter order confusion**:
   - Current: `train_x, train_y, x_train, y_train, batch_size`
   - Unclear what `x_train` represents (should be CNN features)

4. **Type issues**:
   - Function expects numpy arrays but code passes torch tensors

## What the Function Actually Needs:

Based on its purpose (reshaping CNN features to RayBNN format):

### Required Parameters:
1. **`cnn_features`** (numpy array): CNN-extracted features, shape `(N, feature_dim)`
2. **`labels`** (numpy array): Labels, shape `(N,)`
3. **`input_size`** (int): Feature dimension (e.g., 1176)
4. **`output_size`** (int): Number of classes (e.g., 10)
5. **`batch_size`** (int): Batch size for RayBNN (e.g., 1000)
6. **`traj_size`** (int): Trajectory size (usually 1)
7. **`num_samples`** (int): Number of sample batches (e.g., 60)

### Optional Parameters:
- None needed - function should be self-contained

## Recommended Function Signature:

```python
def reshape_cnn_features_to_raybnn_format(
    cnn_features: np.ndarray,      # Shape: (N, feature_dim)
    labels: np.ndarray,             # Shape: (N,)
    input_size: int,                # e.g., 1176
    output_size: int,               # e.g., 10
    batch_size: int,                # e.g., 1000
    traj_size: int = 1,             # Usually 1
    num_samples: int = None         # Auto-calculate if None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape CNN features to RayBNN format.
    
    Args:
        cnn_features: CNN-extracted features, shape (N, feature_dim)
        labels: Labels, shape (N,)
        input_size: Feature dimension
        output_size: Number of output classes
        batch_size: Batch size for RayBNN
        traj_size: Trajectory size (default: 1)
        num_samples: Number of sample batches. If None, calculated from data size.
    
    Returns:
        train_x: RayBNN format array, shape (input_size, batch_size, traj_size, num_samples)
        train_y: RayBNN format array, shape (output_size, batch_size, traj_size, num_samples)
    """
```












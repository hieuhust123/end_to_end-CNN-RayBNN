import torch
import torch.nn as nn
import raybnn_python
import numpy as np
from typing import Tuple

class RayBNNAutograd(torch.autograd.Function):
    """
    Custom PyTorch autograd function for RayBNN forward and backward passes.
    This allows RayBNN to be part of the PyTorch computational graph.
    """
    
    @staticmethod
    def forward(ctx, features: torch.Tensor, arch_search, traj_size: int, max_epoch: int) -> torch.Tensor:
        """
        Forward pass through RayBNN.
        
        Args:
            features: PyTorch tensor of shape [batch_size, feature_dim]
            arch_search: RayBNN model
            traj_size: trajectory size
            max_epoch: maximum epochs
            
        Returns:
            PyTorch tensor output from RayBNN
        """
        # Store inputs for backward pass
        ctx.arch_search = arch_search
        ctx.traj_size = traj_size
        ctx.max_epoch = max_epoch
        ctx.feature_shape = features.shape
        
        # Convert PyTorch tensor to numpy for RayBNN
        features_np = features.detach().cpu().numpy()
        
        # Get batch size and feature dimensions
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        
        # Create training data in the format expected by RayBNN
        # Format: [input_size, batch_size, traj_size, num_trajectories]
        train_x = np.zeros((feature_dim, batch_size, traj_size, 1), dtype=np.float32)
        train_y = np.zeros((10, batch_size, traj_size, 1), dtype=np.float32)  # Assuming 10 classes
        
        # Fill the training data
        for i in range(batch_size):
            train_x[:, i, 0, 0] = features_np[i, :]
            # You might want to pass actual labels here instead of zeros
            # train_y[actual_label, i, 0, 0] = 1.0
        
        # Call RayBNN forward pass using the new PyTorch-compatible function
        output = raybnn_python.state_space_forward_pytorch(
            features, arch_search, traj_size, max_epoch
        )
        
        # Convert output back to PyTorch tensor
        # The output should already be a PyTorch tensor from the Rust function
        output_tensor = torch.from_numpy(output) if isinstance(output, np.ndarray) else output
        
        return output_tensor
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """
        Backward pass through RayBNN.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to input features
        """
        # For now, return a zero gradient
        # You'll need to implement the actual backward pass
        # This requires implementing gradient computation in RayBNN
        grad_input = torch.zeros(ctx.feature_shape, dtype=torch.float32)
        
        return grad_input, None, None, None, None

class RayBNNLayer(nn.Module):
    """
    PyTorch module wrapper for RayBNN.
    """
    
    def __init__(self, arch_search, traj_size: int = 1, max_epoch: int = 100):
        super(RayBNNLayer, self).__init__()
        self.arch_search = arch_search
        self.traj_size = traj_size
        self.max_epoch = max_epoch
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RayBNN layer.
        
        Args:
            features: Input features from CNN
            
        Returns:
            RayBNN output
        """
        return RayBNNAutograd.apply(features, self.arch_search, self.traj_size, self.max_epoch)

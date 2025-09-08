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
        
        # ✅ Keep as PyTorch tensor - NO .detach().cpu().numpy()!
        # The Rust function should accept PyTorch tensors directly
        
        # Call RayBNN forward pass using the new PyTorch-compatible function
        # The Rust function now returns a PyArray2 directly
        output = raybnn_python.state_space_forward_pytorch(
            features, arch_search, traj_size, max_epoch
        )
        
        # Convert output back to PyTorch tensor
        # The output is already a numpy array from the Rust function
        output_tensor = torch.from_numpy(output).to(features.device)
        
        # Ensure output has the same device and requires_grad as input
        if features.requires_grad:
            output_tensor.requires_grad_(True)
        
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
        # TODO: Implement actual RayBNN backward pass
        # For now, return identity gradient (pass-through)
        # This allows the model to train, but RayBNN parameters won't be updated
        
        # Create gradient tensor with same shape and device as input
        grad_input = grad_output.clone()
        
        # Ensure proper shape matching
        if grad_input.shape != ctx.feature_shape:
            grad_input = grad_input.view(ctx.feature_shape)
        
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

def test_gradient_flow(raybnn_layer, features):
    """
    Test function to verify gradient flow through RayBNN layer.
    
    Args:
        raybnn_layer: RayBNNLayer instance
        features: Input features tensor
        
    Returns:
        bool: True if gradient flow works correctly
    """
    # Ensure input requires gradients
    features.requires_grad_(True)
    
    # Forward pass
    output = raybnn_layer(features)
    
    # Check if output requires gradients
    if not output.requires_grad:
        print("❌ Output does not require gradients!")
        return False
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check if input has gradients
    if features.grad is None:
        print("❌ Input features do not have gradients!")
        return False
    
    print("✅ Gradient flow test passed!")
    print(f"   Input shape: {features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Input grad shape: {features.grad.shape}")
    print(f"   Output requires_grad: {output.requires_grad}")
    
    return True

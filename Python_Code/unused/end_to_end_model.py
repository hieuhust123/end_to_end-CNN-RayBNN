import torch
import torch.nn as nn
import torch.nn.functional as F
from raybnn_autograd import RayBNNLayer

class EndToEndModel(nn.Module):
    """
    True end-to-end CNN + RayBNN model with gradient flow.
    """
    
    def __init__(self, arch_search, traj_size: int = 1, max_epoch: int = 100):
        super(EndToEndModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(p=0.2)
        
        # RayBNN layer
        self.raybnn = RayBNNLayer(arch_search, traj_size, max_epoch)
        
        # Optional: Add a final classification layer
        self.classifier = nn.Linear(10, 10)  # Assuming 10 classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire model.
        
        Args:
            x: Input images [batch_size, 1, 28, 28]
            
        Returns:
            Model output [batch_size, num_classes]
        """
        # CNN forward pass
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.drop(self.conv3(x)))
        
        # Flatten features
        features = x.view(x.size(0), -1)  # [batch_size, feature_dim]
        
        # RayBNN forward pass
        raybnn_output = self.raybnn(features)
        
        # Optional: Apply final classification
        output = self.classifier(raybnn_output)
        
        return output
    
    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get CNN features without RayBNN processing.
        Useful for debugging or feature analysis.
        """
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.drop(self.conv3(x)))
        return x.view(x.size(0), -1)
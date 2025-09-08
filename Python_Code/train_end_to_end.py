import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import fetch_openml
from end_to_end_model import EndToEndModel
import raybnn_python

def load_mnist():
    """Load and preprocess MNIST dataset."""
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0
    
    x_train = X[:60000].reshape(-1, 28, 28)
    y_train = y[:60000].astype(np.int64)
    x_test = X[60000:].reshape(-1, 28, 28)
    y_test = y[60000:].astype(np.int64)
    
    # Normalize
    mean_value = np.mean(x_train)
    max_value = np.max(x_train)
    min_value = np.min(x_train)
    
    x_train = (x_train - mean_value) / (max_value - min_value)
    x_test = (x_test - mean_value) / (max_value - min_value)
    
    return x_train, y_train, x_test, y_test

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Convert to PyTorch tensors
    x_train_tensor = torch.from_numpy(x_train).float().unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.from_numpy(y_train).long()
    x_test_tensor = torch.from_numpy(x_test).float().unsqueeze(1)
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # Create RayBNN architecture
    dir_path = "/tmp/"
    max_input_size = 1176  # CNN output size: 24 * 7 * 7
    input_size = 1176
    max_output_size = 10
    output_size = 10
    max_neuron_size = 2000
    batch_size = 1000
    traj_size = 1
    proc_num = 2
    active_size = 1000
    
    arch_search = raybnn_python.create_start_archtecture(
        input_size, max_input_size, output_size, max_output_size,
        active_size, max_neuron_size, batch_size, traj_size, proc_num, dir_path
    )
    
    sphere_rad = arch_search["neural_network"]["netdata"]["sphere_rad"]
    
    arch_search = raybnn_python.add_neuron_to_existing3(
        10, 10000, sphere_rad/1.3, sphere_rad/1.3, sphere_rad/1.3, arch_search
    )
    
    arch_search = raybnn_python.select_forward_sphere(arch_search)
    
    # Create end-to-end model
    model = EndToEndModel(arch_search, traj_size=1, max_epoch=100).to(device)
    
    # Create optimizer for both CNN and RayBNN parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    num_epochs = 10
    batch_size_train = 100
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch}: Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for i in range(0, len(x_test_tensor), batch_size_train):
            batch_data = x_test_tensor[i:i+batch_size_train].to(device)
            batch_target = y_test_tensor[i:i+batch_size_train].to(device)
            
            output = model(batch_data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(batch_target.view_as(pred)).sum().item()
            test_total += batch_target.size(0)
    
    test_accuracy = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()





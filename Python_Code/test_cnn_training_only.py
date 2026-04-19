import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_openml
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# saved as pretrained CNN


def plot_cnn(history, save_path="test_cnn_training_only_20_epochs"):
    train_losses = history['train_losses']
    test_losses = history['test_losses']
    train_accs = history['train_accuracies']
    test_accs = history['test_accuracies']

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12,5))

    # Plot 1: Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, test_losses, 'r--', label='Test Loss')

    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]
    plt.annotate(f'{final_train_loss:.4f}', xy=(epochs[-1], final_train_loss), 
                 xytext=(5, 5), textcoords='offset points', color='blue')
    plt.annotate(f'{final_test_loss:.4f}', xy=(epochs[-1], final_test_loss), 
                 xytext=(5, -15), textcoords='offset points', color='red')

    plt.title('CNN only model cross-entropy loss')
    plt.xlabel('Epochs')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def load_mnist():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0

    x_train = X[:60000].reshape(-1, 28, 28)
    y_train = y[:60000]
    x_test = X[60000:].reshape(-1, 28, 28)
    y_test = y[60000:]

    return x_train, y_train, x_test, y_test

class CNN(nn.Module):
    def __init__(self, num_classes=10, use_dropout=False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, 
                               kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, 
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24,
                               kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(p=0.1)
        self.use_dropout = use_dropout
        self.fc = nn.Linear(24*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.drop(self.conv3(x)))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
    
    def get_features(self, x):
        """Extract features without classification layer"""
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.drop(self.conv3(x)))
        x = F.dropout(x, training=False)  # No dropout during feature extraction
        return x.reshape(x.size(0), -1)

def train_model(model, x_train, y_train, x_test, y_test, batch_size,
num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    model.train()
    num_batches = len(x_train) // batch_size
    print(f"\n{'='*60}")
    print(f"Training CNN on MNIST")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {learning_rate}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        print(f"EPOCH {epoch+1}/{num_epochs}")
        print("-" * 60)

        # Training loop
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()

            # Forward pass
            output = model(batch_x)
            loss = criterion(output, batch_y)
            
            # Backward pass
            loss.backward()

            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            epoch_total += batch_y.size(0)
            epoch_correct += (predicted == batch_y).sum().item()
            epoch_loss += loss.item()

            # Epoch statistics
        avg_train_loss = epoch_loss / num_batches  # Training loss WITH dropout (regularized)
        train_accuracy = epoch_correct / epoch_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluation on test set (without dropout)
        test_loss, test_accuracy = evaluate_model(model, x_test, y_test, 
                                                   batch_size, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Optional: Also compute training loss WITHOUT dropout for fair comparison
        # This shows what the training loss would be without regularization
        # train_loss_no_dropout, _ = evaluate_model(model, x_train, y_train, 
        #                                            batch_size, criterion)
        # print(f"  Train Loss (no dropout): {train_loss_no_dropout:.6f}")
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time

        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
        print(f"  Test Loss: {test_loss:.6f}")
        print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Train Accuracy: {train_accuracies[-1]:.4f} ({train_accuracies[-1]*100:.1f}%)")
    print(f"Final Test Loss: {test_losses[-1]:.6f}")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.4f} ({test_accuracies[-1]*100:.1f}%)")
    print(f"Loss improvement: {train_losses[0] - train_losses[-1]:.6f}")
    print(f"Accuracy improvement: {train_accuracies[-1] - train_accuracies[0]:.4f}")
    print(f"{'='*60}\n")
    
    # Save conv layers
    conv_state = {
        'conv1.weight' : model.conv1.weight,
        'conv1.bias' : model.conv1.bias,
        'conv2.weight' : model.conv2.weight,
        'conv2.bias' : model.conv2.bias,
        'conv3.weight' : model.conv3.weight,
        'conv3.bias' : model.conv3.bias,
    }
    torch.save(conv_state, 'cnn_pretrained_conv_only.pth')
    print("Saved pretrained CNN weights!")

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }

def evaluate_model(model, x_test, y_test, batch_size, criterion):
    """Evaluate the model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        num_batches = len(x_test) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_x = x_test[start_idx:end_idx]
            batch_y = y_test[start_idx:end_idx]
            
            output = model(batch_x)
            loss = criterion(output, batch_y)
            
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    model.train()
    avg_loss = test_loss / num_batches
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    """Main function"""
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Convert labels to integers
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # Normalize data
    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)
    
    x_train = (x_train.astype(np.float32) - mean_value) / (max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value) / (max_value - min_value)
    
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Convert to PyTorch tensors
    x_train_tensor = torch.from_numpy(x_train).float().unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.from_numpy(y_train).long()
    x_test_tensor = torch.from_numpy(x_test).float().unsqueeze(1)
    y_test_tensor = torch.from_numpy(y_test).long()
    
    print(f"x_train_tensor shape: {x_train_tensor.shape}")
    print(f"y_train_tensor shape: {y_train_tensor.shape}")
    
    # Create model
    model = CNN(num_classes=10)
    print(f"\nModel created:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training parameters
    batch_size = 1000
    num_epochs = 1000
    learning_rate = 0.001
    
    # Train model
    history = train_model(
        model, 
        x_train_tensor, 
        y_train_tensor,
        x_test_tensor,
        y_test_tensor,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    plot_cnn(history, save_path="cnn_training_metrics.png")
    

    print("Training completed!")
    return model, history



if __name__ == '__main__':
    model, history = main()
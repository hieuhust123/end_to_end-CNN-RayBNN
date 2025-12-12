import matplotlib.pyplot as plt
import numpy as np

# Read data from plot.txt
epochs = []
train_loss = []
test_loss = []

with open('plot.txt', 'r') as f:
    lines = f.readlines()
    # Skip header lines (first 3 lines)
    for line in lines[3:]:
        line = line.strip()
        if line:  # Skip empty lines
            # Split by whitespace (tabs or spaces) and filter out empty strings
            parts = [p for p in line.split() if p.strip()]
            if len(parts) >= 3:
                try:
                    epochs.append(int(parts[0]))
                    train_loss.append(float(parts[1]))
                    test_loss.append(float(parts[2]))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Parts: {parts}")
                    continue

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=8)
plt.plot(epochs, test_loss, 'r-s', label='Test Loss', linewidth=2, markersize=8)

# Set labels and title
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('RayBNN loss on raw MNIST dataset', fontsize=14, fontweight='bold')

# Set x-axis to show epochs from 0 to 9
plt.xlim(-0.5, 9.5)
plt.xticks(range(0, 10))

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Add legend
plt.legend(loc='best', fontsize=11)

# Add text annotations for last epoch losses
if epochs and train_loss and test_loss:
    last_epoch = epochs[-1]
    last_train_loss = train_loss[-1]
    last_test_loss = test_loss[-1]
    
    # Add text annotations near the last data points
    plt.text(last_epoch + 0.3, last_train_loss, f'Train: {last_train_loss:.4f}', 
             fontsize=10, verticalalignment='center', color='blue', fontweight='bold')
    plt.text(last_epoch + 0.3, last_test_loss, f'Test: {last_test_loss:.4f}', 
             fontsize=10, verticalalignment='center', color='red', fontweight='bold')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('raybnn_training_loss_plot_mnist.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'raybnn_training_loss_plot_mnist.png'")

# Show the plot
plt.show()


import matplotlib.pyplot as plt
import re

# Read data from output_raybnn_training.txt
epochs = []
train_loss = []

with open('output_raybnn_training.txt', 'r') as f:
    for line in f:
        # Look for lines with "Train loss:" pattern
        # Format: "Train loss: 0.969746, alpha0: 0.01, i: 0"
        match = re.search(r'Train loss:\s+([\d.]+).*i:\s+(\d+)', line)
        if match:
            loss_value = float(match.group(1))
            epoch = int(match.group(2))
            epochs.append(epoch)
            train_loss.append(loss_value)

# Sort by epoch to ensure correct order
if epochs:
    sorted_data = sorted(zip(epochs, train_loss))
    epochs, train_loss = zip(*sorted_data)
    epochs = list(epochs)
    train_loss = list(train_loss)
else:
    epochs, train_loss = [], []

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)

# Set labels and title
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('RayBNN Training loss on MNIST dataset with CNN features extraction', fontsize=14, fontweight='bold')

# Set x-axis range/ticks based on data size
if epochs:
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    plt.xlim(min_epoch - 0.5, max_epoch + 0.5)

    epoch_count = len(epochs)
    if epoch_count <= 20:
        plt.xticks(epochs)
    else:
        step = max(1, epoch_count // 10)
        xticks = list(range(min_epoch, max_epoch + 1, step))
        if xticks[-1] != max_epoch:
            xticks.append(max_epoch)
        plt.xticks(xticks)

# Set y-axis scale based on the data with a small margin
if train_loss:
    y_min = min(train_loss)
    y_max = max(train_loss)
    margin = max(0.01, (y_max - y_min) * 0.1)
    plt.ylim(y_min - margin, y_max + margin)
    # plt.ylim(0.2,1.2)
    # plt.yticks([0.2,0.4,0.6,0.8,1.0,1.2])

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Add legend
plt.legend(loc='best', fontsize=11)

# Add text annotation for last epoch loss
if epochs and train_loss:
    last_epoch = epochs[-1]
    last_train_loss = train_loss[-1]
    
    # Add text annotation near the last data point
    plt.text(last_epoch + 0.3, last_train_loss, f'Train: {last_train_loss:.4f}', 
             fontsize=10, verticalalignment='center', color='blue', fontweight='bold')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('raybnn_training_loss_from_cnn_features.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'raybnn_training_loss_from_cnn_features.png'")
print(f"Plotted {len(epochs)} epochs: {epochs}")
print(f"Loss values: {train_loss}")

# Show the plot
plt.show()


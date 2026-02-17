import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

# Read the file
with open('alpha_0.001_performance_raybnn.txt', 'r') as f:
    content = f.read()

# Extract train loss values
pattern = r'Train loss: ([\d.]+), alpha0: ([\d.]+), i: (\d+)'
matches = re.findall(pattern, content)

iterations = []
losses = []

for match in matches:
    loss = float(match[0])
    iteration = int(match[2])
    iterations.append(iteration)
    losses.append(loss)

# Extract accuracy
accuracy_pattern = r'Accuracy:\s+([\d.]+)'
accuracy_match = re.search(accuracy_pattern, content)
accuracy = float(accuracy_match.group(1)) if accuracy_match else None

# Apply smoothing
window_size = 10
smoothed_losses = uniform_filter1d(losses, size=window_size, mode='nearest')

# Create the plot
fig, ax = plt.subplots(figsize=(14, 7))

# Plot raw data
ax.plot(iterations, losses, linewidth=0.5, color='lightblue', 
        alpha=0.5, label='Raw Loss')

# Plot smoothed data
ax.plot(iterations, smoothed_losses, linewidth=2, color='darkblue', 
        label=f'Smoothed Loss (window={window_size})')

ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
ax.set_title('RayBNN Training Loss (lr=0.001)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, max(iterations))

# Add statistics box with accuracy
textstr = f'Initial Loss: {losses[0]:.4f}\nFinal Loss: {losses[-1]:.4f}\nMin Loss: {min(losses):.4f}\nIterations: {len(iterations)}'
if accuracy is not None:
    textstr += f'\nAccuracy: {accuracy:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('training_loss_detailed.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'training_loss_detailed.png'")
plt.show()
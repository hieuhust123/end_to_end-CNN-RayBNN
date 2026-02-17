#!/usr/bin/env python3
"""
Script to plot training metrics from output log files.
Usage: python plot_training_results.py <log_file>
"""

import re
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import sys

def extract_metrics(log_file):
    """Extract training metrics from log file and aggregate by epoch"""
    epoch_losses = []
    epoch_accuracies = []
    
    current_epoch_losses = []
    current_epoch_accs = []
    
    raybnn_losses = []
    raybnn_iterations = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Detect epoch boundaries
            epoch_match = re.search(r"EPOCH (\d+)/\d+", line)
            if epoch_match:
                # Save previous epoch's data
                if current_epoch_losses:
                    epoch_losses.append(np.mean(current_epoch_losses))
                    epoch_accuracies.append(np.mean(current_epoch_accs))
                    current_epoch_losses = []
                    current_epoch_accs = []
            
            # Extract batch metrics (format: Batch X/Y: loss=Z, acc=A, entropy=B)
            batch_match = re.search(r"Batch \d+/\d+: loss=([\d\.]+), acc=([\d\.]+)", line)
            if batch_match:
                loss = float(batch_match.group(1))
                acc = float(batch_match.group(2))
                current_epoch_losses.append(loss)
                current_epoch_accs.append(acc)
            
            # Extract RayBNN training loss
            raybnn_match = re.search(r"Train loss:\s*([\d\.]+)", line)
            if raybnn_match:
                raybnn_losses.append(float(raybnn_match.group(1)))
                raybnn_iterations.append(len(raybnn_losses))
    
    # Don't forget the last epoch
    if current_epoch_losses:
        epoch_losses.append(np.mean(current_epoch_losses))
        epoch_accuracies.append(np.mean(current_epoch_accs))
    
    return {
        'epoch_losses': epoch_losses,
        'epoch_accuracies': epoch_accuracies,
        'raybnn_losses': raybnn_losses,
        'raybnn_iterations': raybnn_iterations
    }

def plot_training_loss(metrics, output_file='training_loss.png'):
    """Plot training loss by epoch"""
    if not metrics['epoch_losses']:
        print("⚠ No training loss data found in log file")
        return
    
    plt.figure(figsize=(12, 7))
    
    epochs = range(1, len(metrics['epoch_losses']) + 1)
    
    # Plot loss
    plt.plot(epochs, metrics['epoch_losses'], 'o-', color='tab:blue', 
             linewidth=2.5, markersize=8, label='Training Loss', alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training Loss', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics box
    initial = metrics['epoch_losses'][0]
    final = metrics['epoch_losses'][-1]
    min_loss = min(metrics['epoch_losses'])
    mean_loss = np.mean(metrics['epoch_losses'])
    
    stats_text = f"Initial: {initial:.4f}\nFinal: {final:.4f}\n"
    stats_text += f"Min: {min_loss:.4f}\nMean: {mean_loss:.4f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # Add mean line
    plt.axhline(y=mean_loss, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'Mean: {mean_loss:.4f}')
    
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Training loss plot saved to: {output_file}")
    plt.close()

def plot_training_accuracy(metrics, output_file='training_accuracy.png'):
    """Plot training accuracy by epoch"""
    if not metrics['epoch_accuracies']:
        print("⚠ No training accuracy data found in log file")
        return
    
    plt.figure(figsize=(12, 7))
    
    epochs = range(1, len(metrics['epoch_accuracies']) + 1)
    
    # Plot accuracy
    plt.plot(epochs, metrics['epoch_accuracies'], 's-', color='tab:green',
             linewidth=2.5, markersize=8, label='Training Accuracy', alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Training Accuracy', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Auto-scale y-axis if accuracy variation is small
    acc_range = max(metrics['epoch_accuracies']) - min(metrics['epoch_accuracies'])
    if acc_range < 0.1:
        margin = max(0.02, acc_range * 0.3)
        min_acc = min(metrics['epoch_accuracies'])
        max_acc = max(metrics['epoch_accuracies'])
        plt.ylim([max(0, min_acc - margin), min(1.0, max_acc + margin)])
    else:
        plt.ylim([0, 1.05])
    
    # Add statistics box
    initial = metrics['epoch_accuracies'][0]
    final = metrics['epoch_accuracies'][-1]
    max_acc = max(metrics['epoch_accuracies'])
    mean_acc = np.mean(metrics['epoch_accuracies'])
    
    stats_text = f"Initial: {initial:.4f}\nFinal: {final:.4f}\n"
    stats_text += f"Max: {max_acc:.4f}\nMean: {mean_acc:.4f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Training accuracy plot saved to: {output_file}")
    plt.close()

def print_summary(metrics):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    
    if metrics['epoch_losses'] and metrics['epoch_accuracies']:
        print(f"\n📊 Epoch Metrics:")
        print(f"  Total Epochs: {len(metrics['epoch_losses'])}")
        print(f"  Initial Loss: {metrics['epoch_losses'][0]:.4f}")
        print(f"  Final Loss: {metrics['epoch_losses'][-1]:.4f}")
        print(f"  Min Loss: {min(metrics['epoch_losses']):.4f}")
        print(f"  Mean Loss: {np.mean(metrics['epoch_losses']):.4f}")
        print()
        print(f"  Initial Accuracy: {metrics['epoch_accuracies'][0]:.4f}")
        print(f"  Final Accuracy: {metrics['epoch_accuracies'][-1]:.4f}")
        print(f"  Max Accuracy: {max(metrics['epoch_accuracies']):.4f}")
        print(f"  Mean Accuracy: {np.mean(metrics['epoch_accuracies']):.4f}")
        
        # Learning progress analysis
        loss_change = metrics['epoch_losses'][0] - metrics['epoch_losses'][-1]
        acc_change = metrics['epoch_accuracies'][-1] - metrics['epoch_accuracies'][0]
        
        print(f"\n📈 Learning Progress:")
        print(f"  Loss Change: {loss_change:+.4f} ({loss_change/metrics['epoch_losses'][0]*100:+.1f}%)")
        print(f"  Accuracy Change: {acc_change:+.4f} ({acc_change*100:+.1f} percentage points)")
        
        if abs(loss_change) < 0.001:
            print(f"  ⚠ WARNING: Loss is not changing (model may not be learning)")
        elif loss_change > 0:
            print(f"  ✓ Loss is decreasing (good)")
        else:
            print(f"  ✗ Loss is increasing (bad)")
    
    if metrics['raybnn_losses']:
        print(f"\n🔬 RayBNN Training:")
        print(f"  Total Iterations: {len(metrics['raybnn_losses'])}")
        print(f"  Initial Loss: {metrics['raybnn_losses'][0]:.6f}")
        print(f"  Final Loss: {metrics['raybnn_losses'][-1]:.6f}")
        print(f"  Min Loss: {min(metrics['raybnn_losses']):.6f}")
    
    print("="*60 + "\n")

def main():
    # Get log file from command line or use default
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "output_cnn+raybnn_training_784.txt"
    
    print(f"Reading metrics from: {log_file}")
    
    try:
        # Extract metrics
        metrics = extract_metrics(log_file)
        
        # Create separate plots
        plot_training_loss(metrics, 'training_loss.png')
        plot_training_accuracy(metrics, 'training_accuracy.png')
        
        # Print summary
        print_summary(metrics)
        
        print("✓ All plots generated successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{log_file}' not found!")
        print(f"Usage: python {sys.argv[0]} <log_file>")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

import re
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def parse_losses_from_log(log_file):
    """
    Parse loss values from the log file.
    Returns:
        batch_losses: list of batch-level losses
        epoch_losses: list of epoch-level losses
    """
    batch_losses = []
    epoch_losses = []
    
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found!")
        return batch_losses, epoch_losses
    
    try:
        in_gradient_summary = False
        with open(log_file, 'r') as f:
            for line in f:
                # Match batch-level losses: "  Batch 0: loss=2.3026, acc=0.088"
                batch_match = re.search(r"Batch\s+\d+:\s+loss=([\d\.]+)", line)
                if batch_match:
                    batch_losses.append(float(batch_match.group(1)))
                
                # Track if we're in a GRADIENT FLOW SUMMARY section
                if "GRADIENT FLOW SUMMARY" in line:
                    in_gradient_summary = True
                    continue
                
                # Match epoch-level losses: "  Loss: 2.302576 (change: N/A)"
                # These appear right after "GRADIENT FLOW SUMMARY"
                if in_gradient_summary:
                    epoch_match = re.search(r"^\s+Loss:\s+([\d\.]+)", line)
                    if epoch_match:
                        epoch_losses.append(float(epoch_match.group(1)))
                        in_gradient_summary = False
                    elif line.strip() == "" or "EPOCH" in line:
                        # Reset if we hit a blank line or new epoch before finding Loss
                        in_gradient_summary = False
    
    except Exception as e:
        print(f"Error reading log file: {e}")
        return batch_losses, epoch_losses
    
    return batch_losses, epoch_losses

def plot_losses(batch_losses, epoch_losses, output_dir="."):
    """
    Plot batch-level and epoch-level losses.
    """
    # Plot batch-level losses
    if batch_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(batch_losses, label='Batch Loss', alpha=0.6, linewidth=0.5)
        plt.xlabel('Batch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Batch-Level Training Loss of model when train CNN and RayBNN(lr=0.0001-0.001)')
        final_batch_loss = batch_losses[-1]
        plt.text(0.02, 0.95, f'Final batch loss: {final_batch_loss:.4f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.legend()
        plt.grid(True)
        batch_plot_filename = os.path.join(output_dir, "cnn_batch_freeze_cnn_train_raybnn.png")
        plt.savefig(batch_plot_filename)
        print(f"Batch-level loss plot saved to {batch_plot_filename}")
        plt.close()
    
    # Plot epoch-level losses
    if epoch_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses, label='Epoch Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss of model when train CNN and RayBNN(lr=0.0001-0.001)')
        # Set x-axis ticks to integer values (spacing of 1)
        max_epoch = len(epoch_losses) - 1
        plt.xticks(np.arange(0, max_epoch + 1, 1))
        final_epoch_loss = epoch_losses[-1]
        plt.text(0.02, 0.95, f'Final loss: {final_epoch_loss:.4f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.legend()
        plt.grid(True)
        epoch_plot_filename = os.path.join(output_dir, "plot_train_cnn_train_raybnn.png")
        plt.savefig(epoch_plot_filename)
        print(f"Epoch-level loss plot saved to {epoch_plot_filename}")
        plt.close()

def main():
    log_file = "output_cnn+raybnn_training.txt"
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    print(f"Parsing losses from: {log_file}")
    batch_losses, epoch_losses = parse_losses_from_log(log_file)
    
    print(f"Found {len(batch_losses)} batch-level losses")
    print(f"Found {len(epoch_losses)} epoch-level losses")
    
    if not batch_losses and not epoch_losses:
        print("No losses found in the log file!")
        return
    
    if batch_losses:
        print(f"Batch loss range: [{min(batch_losses):.4f}, {max(batch_losses):.4f}]")
        print(f"Final batch loss: {batch_losses[-1]:.4f}")
    
    if epoch_losses:
        print(f"Epoch loss range: [{min(epoch_losses):.4f}, {max(epoch_losses):.4f}]")
        print(f"Final epoch loss: {epoch_losses[-1]:.4f}")
    
    plot_losses(batch_losses, epoch_losses)
    print("Done!")

if __name__ == '__main__':
    main()


import numpy as np
import raybnn_python
import torch 
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import os
from torchvision import datasets, transforms,utils
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torch import optim
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        latent_shape = self._get_conv_shape(input_dim=input_dim)
        self.fc1 = nn.Linear(latent_shape[1], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5) 

    def _get_conv_shape(self, input_dim):
        zeros = torch.zeros(input_dim).unsqueeze(0)
        x = self.conv1(zeros)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x.shape

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        features = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))   
        x = self.fc3(x)  
        return x, features

def plot_cnn_training(train_losses, train_accs, test_losses, test_accs, fold=None):
    """Plot CNN training metrics."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title_suffix = "" if fold is None else f" - Fold {fold}"
    fig.suptitle(f'CNN Training Metrics{title_suffix}', fontsize=14)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if test_losses:
        axes[0].plot(epochs, test_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    if test_accs:
        axes[1].plot(epochs, test_accs, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    fold_suffix = "" if fold is None else f"_fold{fold}"
    plt.savefig(f'Xuan_Chen_cnn_training{fold_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'CNN training plot saved to: Xuan_Chen_cnn_training{fold_suffix}.png')

def plot_kfold_summary(accuracies, precisions, recalls, f1s):
    """Plot summary of K-fold cross-validation results."""
    n_folds = len(accuracies)
    folds = range(1, n_folds + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_folds)
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1s, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('10-Fold Cross-Validation Results (CNN)')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add statistics text
    stats_text = (f"Avg Acc: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n"
                  f"Avg F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('kfold_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('K-fold summary plot saved to: kfold_summary.png')

def plot_raybnn_training(accuracies, precisions, recalls, f1s, losses):
    """Plot RayBNN training metrics over epochs."""
    epochs = range(1, len(accuracies) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RayBNN Training Metrics', fontsize=14)
    
    # Accuracy
    axes[0, 0].plot(epochs, accuracies, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy over Epochs')
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim([0, 1])
    
    # Precision, Recall, F1
    axes[0, 1].plot(epochs, precisions, 'g-', label='Precision', linewidth=2)
    axes[0, 1].plot(epochs, recalls, 'r-', label='Recall', linewidth=2)
    axes[0, 1].plot(epochs, f1s, 'm-', label='F1 Score', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision/Recall/F1 over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([0, 1])
    
    # Loss
    if losses:
        axes[1, 0].plot(range(1, len(losses)+1), losses, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss over Epochs')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'Loss data not available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Loss (unavailable)')
    
    # Summary statistics
    summary_text = f"""Final Metrics:
    Accuracy:  {accuracies[-1]:.4f}
    Precision: {precisions[-1]:.4f}
    Recall:    {recalls[-1]:.4f}
    F1 Score:  {f1s[-1]:.4f}

Best Metrics:
    Max Accuracy:  {max(accuracies):.4f} (Epoch {accuracies.index(max(accuracies))+1})
    Max F1:        {max(f1s):.4f} (Epoch {f1s.index(max(f1s))+1})"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                    verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('raybnn_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('RayBNN training plot saved to: raybnn_training.png')

def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    data_root = "home/hbui/Downloads/xuanchen_code/c/mnist/"

    full_dataset = ConcatDataset([
        datasets.MNIST(root=data_root, transform=transform, train=True, download=True),
        datasets.MNIST(root=data_root, transform=transform, train=False, download=True)
    ])
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies= []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)

        net = CNN((1, 28, 28), 10, 128)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
        epoch_train_loss = []
        epoch_train_accs = []
        epoch_test_loss = []
        epoch_test_accs = []
        test_accs = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        
        n_ep=20

        for epoch in range(n_ep):
            net.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, features = net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss.append(running_loss / len(train_loader))
            epoch_train_accs.append(100 * correct_train / total_train)
            print(f'Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.5f}, Accuracy: {100*correct_train/total_train:.5f}%')
            
            all_features = []

            correct = 0
            total = 0
            test_features = []
            test_labels = []
            predicted_labels = []
            true_labels=[]
            net.eval()
            test_loss = 0.0
            correct_test = 0
            total_test = 0
            net.eval()
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, features = net(inputs)
                    loss = loss_fn(outputs, labels)
                    test_loss += loss.item()
                    features = features.cpu().numpy()
                    test_features.append(features)
                    test_labels.append(labels.cpu().numpy())
                    
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_labels.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            # epoch_test_loss.append(test_loss / len(val_loader))
            # epoch_test_accs.append(100 * correct_test / total_test)
         # Extract features from the training set
         
        train_features = []
        train_labels = []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                _, features = net(inputs)
                train_features.append(features.cpu().numpy())
                train_labels.append(labels.numpy())
        train_features = np.vstack(train_features)
        train_labels = np.concatenate(train_labels)

        # Extract features from the validation set
        val_features = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                _, features = net(inputs)
                val_features.append(features.cpu().numpy())
                val_labels.append(labels.numpy())
        val_features = np.vstack(val_features)
        val_labels = np.concatenate(val_labels)
        
        accuracy = accuracy_score(val_labels, output_y)  
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1_scores.append(f1)
        print(f'Fold {fold + 1}: Accuracy={accuracy:.5f}, Precision={precision:.5f}, Recall={recall:.5f}, F1 Score={f1:.5f}')
        print(f'Precision: {precision:.5f}, Recall: {recall:.5f}, F1 Score: {f1:.5f}')
        print(f'Epoch [{epoch+1}/{n_ep}], Test Loss: {test_loss/len(val_loader):.5f}, Test Accuracy: {100*correct_test/total_test:.5f}%')

    train_iters = range(len(epoch_train_accs))

    avg_accuracy = np.mean(fold_accuracies)
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)
    avg_f1_score = np.mean(fold_f1_scores)
    print(f'Average: Accuracy={avg_accuracy:.5f}, Precision={avg_precision:.5f}, Recall={avg_recall:.5f}, F1 Score={avg_f1_score:.5f}')
             
                
    outputs = outputs.cpu().numpy()
    plot_kfold_summary(fold_accuracies, fold_precisions, fold_recalls,
fold_f1_scores)
    return val_features, val_labels, train_features, train_labels

def train_raybnn(x_train, y_train, x_test, y_test):
    accuracy_values = []
    precision_values=[]
    recall_values = []
    f1_values=[]
    loss_values = []
    if isinstance(x_train, torch.Tensor):
        Rey_train = x_train.cpu().numpy()

    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value) / (max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value) / (max_value - min_value)

    dir_path = "/tmp/"

    max_input_size = 512
    input_size = 512

    max_output_size = 10
    output_size = 10

    max_neuron_size = 1000

    batch_size = 1000
    traj_size = 1

    proc_num = 2
    active_size = 1000

    training_samples = x_train.shape[0]
    crossval_samples = x_train.shape[0]
    testing_samples = x_test.shape[0]

    # Format MNIST dataset
    train_x = np.zeros((input_size, batch_size, traj_size, training_samples)).astype(np.float32)
    train_y = np.zeros((output_size, batch_size, traj_size, training_samples)).astype(np.float32)

    for i in range(x_train.shape[0]):
        j = (i % batch_size)
        k = int(i / batch_size)

        train_x[:, j, 0, k] = x_train[i, :]

        idx = y_train[i]
        train_y[idx, j, 0, k] = 1.0

    crossval_x = np.copy(train_x)
    crossval_y = np.copy(train_y)

    # Create Neural Network
    arch_search = raybnn_python.create_start_archtecture(
        input_size,
        max_input_size,

        output_size,
        max_output_size,

        active_size,
        max_neuron_size,

        batch_size,
        traj_size,

        proc_num,
        dir_path
    )

    sphere_rad = arch_search["neural_network"]["netdata"]["sphere_rad"]

    arch_search = raybnn_python.add_neuron_to_existing3(
        10,
        10000,
        sphere_rad / 1.3,
        sphere_rad / 1.3,
        sphere_rad / 1.3,

        arch_search,
    )

    arch_search = raybnn_python.select_forward_sphere(arch_search)

    raybnn_python.print_model_info(arch_search)

    stop_strategy = "STOP_AT_TRAIN_LOSS"
    lr_strategy = "SHUFFLE_CONNECTIONS"
    lr_strategy2 = "MAX_ALPHA"

    loss_function = "sigmoid_cross_entropy_5"

    max_epoch = 0
    stop_epoch = 100000
    stop_train_loss = 0.005

    max_alpha = 0.01

    exit_counter_threshold = 100000
    shuffle_counter_threshold = 200

    total_epochs = 100

    for epoch in range(total_epochs):
        max_epoch += 1
        # Train Neural Network
        arch_search = raybnn_python.train_network(
            train_x,
            train_y,

            crossval_x,
            crossval_y,

            stop_strategy,
            lr_strategy,
            lr_strategy2,

            loss_function,

            max_epoch + 1,
            stop_epoch + 1,
            stop_train_loss,

            max_alpha,

            exit_counter_threshold,
            shuffle_counter_threshold,

            arch_search
        )

        # ADD: Extract training loss if available
        if "training_loss" in arch_search:
            loss_values.append(arch_search["training_loss"])
        
        test_x = np.zeros((input_size, batch_size, traj_size, testing_samples)).astype(np.float32)

        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i / batch_size)

            test_x[:, j, 0, k] = x_test[i, :]

        # Test Neural Network
        output_y = raybnn_python.test_network(
            test_x,

            arch_search
        )

        print(output_y.shape)

        pred = []
        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i / batch_size)

            sample = output_y[:, j, 0, k]
            #print(sample)

            pred.append(np.argmax(sample))

        pred = [np.argmax(output_y[:, i % batch_size, 0, int(i/batch_size)]) for i in range(x_test.shape[0])]
        
        acc = accuracy_score(y_test, pred)
        ret = precision_recall_fscore_support(y_test, pred, average='macro')

        accuracy_values.append(acc)
        precision_values.append(ret[0])
        recall_values.append(ret[1])
        f1_values.append(ret[2])

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'RayBNN Epoch [{epoch+1}/{total_epochs}], '
                  f'Accuracy: {acc:.4f}, Precision: {ret[0]:.4f}, '
                  f'Recall: {ret[1]:.4f}, F1: {ret[2]:.4f}')

    # ADD: Plot RayBNN training metrics
    plot_raybnn_training(accuracy_values, precision_values, recall_values, 
                         f1_values, loss_values, fold=None)

    return output_y.reshape(-1)

# ADD: New plotting function for RayBNN
def plot_raybnn_training(accuracies, precisions, recalls, f1s, losses, fold=None):
    """Plot RayBNN training metrics over epochs."""
    epochs = range(1, len(accuracies) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'RayBNN Training Metrics{"" if fold is None else f" - Fold {fold}"}', 
                 fontsize=16)
    
    # Accuracy
    axes[0, 0].plot(epochs, accuracies, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy over Epochs')
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim([0, 1])
    
    # Precision, Recall, F1
    axes[0, 1].plot(epochs, precisions, 'g-', label='Precision', linewidth=2)
    axes[0, 1].plot(epochs, recalls, 'r-', label='Recall', linewidth=2)
    axes[0, 1].plot(epochs, f1s, 'm-', label='F1 Score', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision/Recall/F1 over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([0, 1])
    
    # Loss (if available)
    if losses:
        axes[1, 0].plot(epochs, losses, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss over Epochs')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'Loss data not available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Loss (unavailable)')
    
    # Summary statistics
    summary_text = f"""Final Metrics:
    Accuracy:  {accuracies[-1]:.4f}
    Precision: {precisions[-1]:.4f}
    Recall:    {recalls[-1]:.4f}
    F1 Score:  {f1s[-1]:.4f}
    
    Best Metrics:
    Max Accuracy:  {max(accuracies):.4f} (Epoch {accuracies.index(max(accuracies))+1})
    Max Precision: {max(precisions):.4f} (Epoch {precisions.index(max(precisions))+1})
    Max Recall:    {max(recalls):.4f} (Epoch {recalls.index(max(recalls))+1})
    Max F1:        {max(f1s):.4f} (Epoch {f1s.index(max(f1s))+1})
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                    verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    fold_suffix = "" if fold is None else f"_fold{fold}"
    plt.savefig(f'run_network_CNN-RayBNN{fold_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'run_network_CNN-RayBNN training plot saved to: run_network_CNN-RayBNN{fold_suffix}.png')


if __name__ == '__main__':

    val_features, val_labels, train_features, train_labels = main()

    output_y = train_raybnn(train_features, train_labels, val_features, val_labels)
    






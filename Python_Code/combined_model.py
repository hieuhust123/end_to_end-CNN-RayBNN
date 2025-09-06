import numpy as np
import raybnn_python
import mnist
import os 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# PyTorch imports for CNN
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Create a CNN class for feature extraction
class CNN_FeatureExtractor(nn.Module):
    # Constructor
    def __init__(self, num_classes=10):
        super(CNN_FeatureExtractor, self).__init__()
        
        # Our images are grayscale, so input channels = 1. We'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # A second convolutional layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # A third convolutional layer takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        
        # Fully-connected layer (not used for feature extraction)
        self.fc = nn.Linear(in_features=24 * 7 * 7, out_features=num_classes)

    def forward(self, x, verbose=False):
        # Use a relu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))
        if verbose:
            print("After conv1 + pool:", x.shape)

        # Use a relu activation function after layer 2 (convolution 2 and pool)
        x = F.relu(self.pool(self.conv2(x)))
        if verbose:
            print("After conv2 + pool:", x.shape)
            
        # Select some features to drop after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(self.conv3(x)))
        if verbose:
            print("After conv3 + drop:", x.shape)
            
        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)
        if verbose:
            print("After dropout:", x.shape)

        # Return the feature maps (without flattening)
        return x

# Function to load Fashion MNIST dataset using PyTorch
data_path = r"D:\Research\Research Fall 2024\ml-basics\fashion"

def load_dataset():
    # Define transformations
    transformation = transforms.Compose([
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load Fashion MNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transformation
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transformation
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

# Function to extract features using the CNN
def extract_cnn_features(model, data_loader, max_batches=None, verbose=False):
    features_list = []
    labels_list = []
    
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if verbose and batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}")
                
            # Forward pass to get features
            features = model(data, verbose=(batch_idx==0 and verbose))
            
            # Reshape features: [batch_size, channels, height, width] -> [batch_size, channels*height*width]
            features = features.reshape(features.size(0), -1)
            
            # Convert to numpy and store
            features_np = features.cpu().numpy()
            labels_np = target.cpu().numpy()
            
            features_list.append(features_np)
            labels_list.append(labels_np)
            
            if max_batches is not None and batch_idx >= max_batches - 1:
                break
    
    # Combine all batches
    features_all = np.vstack(features_list)
    labels_all = np.concatenate(labels_list)
    
    return features_all, labels_all

def main():
    print("Step 1: Loading and preparing Fashion MNIST dataset")
    train_loader, test_loader = load_dataset()
    
    # Fashion MNIST class names
    fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print("Step 2: Creating CNN feature extractor")
    cnn_model = CNN_FeatureExtractor()
    
    print("Step 3: Extracting features from Fashion MNIST using CNN")
    # Extract features for a subset of the data to keep memory usage reasonable
    # Adjust max_batches as needed
    train_features, train_labels = extract_cnn_features(cnn_model, train_loader, max_batches=60, verbose=True)
    test_features, test_labels = extract_cnn_features(cnn_model, test_loader, max_batches=10, verbose=True)
    
    print(f"Extracted features shapes: Train {train_features.shape}, Test {test_features.shape}")
    print(f"Labels shapes: Train {train_labels.shape}, Test {test_labels.shape}")
    
    # Feature dimension from CNN (24*7*7 = 1176)
    feature_dim = train_features.shape[1]
    
    print("Step 4: Setting up RayBNN network parameters")
    dir_path = "/tmp/"
    
    # Update input size to match CNN feature dimension
    max_input_size = feature_dim
    input_size = feature_dim
    
    max_output_size = 10
    output_size = 10
    
    max_neuron_size = 2000
    
    batch_size = 1000
    traj_size = 1
    
    proc_num = 2
    active_size = 1000
    
    # These should match the number of batches we extracted features from
    training_samples = min(60, len(train_features) // batch_size + 1)
    crossval_samples = training_samples
    testing_samples = min(10, len(test_features) // batch_size + 1)
    
    print(f"Using {training_samples} training samples, {testing_samples} testing samples")
    
    print("Step 5: Formatting data for RayBNN")
    # Format train data for RayBNN
    train_x = np.zeros((input_size, batch_size, traj_size, training_samples)).astype(np.float32)
    train_y = np.zeros((output_size, batch_size, traj_size, training_samples)).astype(np.float32)
    
    for i in range(min(train_features.shape[0], batch_size * training_samples)):
        j = (i % batch_size)
        k = int(i / batch_size)
        
        if k >= training_samples:
            break
            
        # Copy CNN features to RayBNN input format
        train_x[:, j, 0, k] = train_features[i, :]
        
        # One-hot encode the labels
        idx = train_labels[i]
        train_y[idx, j, 0, k] = 1.0
    
    # Use the same data for cross-validation for simplicity
    crossval_x = np.copy(train_x)
    crossval_y = np.copy(train_y)
    
    print("Step 6: Creating RayBNN neural network")
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
        sphere_rad/1.3,
        sphere_rad/1.3,
        sphere_rad/1.3,
        
        arch_search,
    )
    
    arch_search = raybnn_python.select_forward_sphere(arch_search)
    
    raybnn_python.print_model_info(arch_search)
    
    print("Step 7: Training RayBNN neural network")
    stop_strategy = "STOP_AT_TRAIN_LOSS"
    lr_strategy = "SHUFFLE_CONNECTIONS"
    lr_strategy2 = "MAX_ALPHA"
    
    loss_function = "sigmoid_cross_entropy_5"
    
    max_epoch = 100000
    stop_epoch = 100000
    stop_train_loss = 0.005
    
    max_alpha = 0.01
    
    exit_counter_threshold = 100000
    shuffle_counter_threshold = 200
    
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
        
        max_epoch,
        stop_epoch,
        stop_train_loss,
        
        max_alpha,
        
        exit_counter_threshold,
        shuffle_counter_threshold,
        
        arch_search
    )
    
    print("Step 8: Preparing test data for RayBNN")
    # Format test data for RayBNN
    test_x = np.zeros((input_size, batch_size, traj_size, testing_samples)).astype(np.float32)
    
    for i in range(min(test_features.shape[0], batch_size * testing_samples)):
        j = (i % batch_size)
        k = int(i / batch_size)
        
        if k >= testing_samples:
            break
            
        # Copy CNN features to RayBNN input format
        test_x[:, j, 0, k] = test_features[i, :]
    
    print("Step 9: Testing RayBNN neural network")
    # Test Neural Network
    output_y = raybnn_python.test_network(
        test_x,
        arch_search
    )
    
    print(f"Output shape: {output_y.shape}")
    
    # Process predictions
    pred = []
    for i in range(min(test_features.shape[0], batch_size * testing_samples)):
        j = (i % batch_size)
        k = int(i / batch_size)
        
        if k >= testing_samples:
            break
            
        sample = output_y[:, j, 0, k]
        pred.append(np.argmax(sample))
    
    # Evaluate predictions
    actual_labels = test_labels[:len(pred)]
    acc = accuracy_score(actual_labels, pred)
    
    ret = precision_recall_fscore_support(actual_labels, pred, average='macro')
    
    print(f"Accuracy: {acc}")
    print(f"Precision, Recall, F1-Score: {ret}")
    
    # # Optional: Visualize some results
    # print("Step 10: Visualizing some results")
    # plt.figure(figsize=(10, 5))
    
    # # Plot 5 test images with predictions
    # for i in range(5):
    #     if i < len(pred):
    #         # Get original image from test_loader
    #         plt.subplot(1, 5, i+1)
    #         img_idx = i
    #         for batch_idx, (data, _) in enumerate(test_loader):
    #             if batch_idx * data.shape[0] <= img_idx < (batch_idx + 1) * data.shape[0]:
    #                 img = data[img_idx - batch_idx * data.shape[0]].squeeze().numpy()
    #                 break
            
    #         plt.imshow(img, cmap='gray')
    #         plt.title(f"Pred: {fashion_mnist_classes[pred[i]]}\nTrue: {fashion_mnist_classes[actual_labels[i]]}")
    #         plt.axis('off')
    
    # plt.tight_layout()
    # plt.savefig('results.png')
    # print("Results visualization saved to 'results.png'")

if __name__ == '__main__':
    main()

import numpy as np
import raybnn_python
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support
from matplotlib import pyplot as plt


class CNN(nn.Module):
	def __init__(self, input_channels=1, num_classes=10, input_size=28):
		super(CNN, self).__init__()
		self.input_size = input_size

		self.features = nn.Sequential(
			# Block 1
			nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(32),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(32),
			nn.MaxPool2d(2, 2),
			nn.Dropout(p=0.3),
			
			# Block 2
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(2, 2),
			nn.Dropout(p=0.3)
		)
		
		self.latent_dim = self._get_conv_output_size((input_channels, input_size, input_size))

		self.classifier = nn.Sequential(
			nn.Linear(self.latent_dim, 256),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.4),
			nn.Linear(256, num_classes)
		)

	def _get_conv_output_size(self, shape):
		"""Calculate feature map size after convolutions"""
		with torch.no_grad():
			dummy_input = torch.zeros(1, *shape)
			output = self.features(dummy_input)
			return int(torch.prod(torch.tensor(output.shape[1:])))

	def forward(self, x, return_features=False):
		"""
		Forward pass with optional feature extraction
		Args:
			x: Input tensor
			return_features: If True, return (predictions, features from 256-dim layer)
		"""
		conv_features = self.features(x)
		conv_features_flat = conv_features.view(conv_features.size(0), -1)
		
		# Get 256-dim features from first FC layer
		# 2. First FC layer: 3136 → 256
		x = self.classifier[0](conv_features_flat)  # Linear
		x = self.classifier[1](x)  # ReLU
		features_256 = x  # ✅ Extract 256-dim features HERE
		
		# 3. Final classification: 256 → 10
		x = self.classifier[2](x)  # Dropout
		predictions = self.classifier[3](x)  # Linear linear layer
		
		if return_features:
			return predictions, features_256  # Return 256-dim features
		return predictions


class CNNTrainer:
	"""Training class for flexibility and reusability"""
	
	def __init__(self, model, device='cuda', lr=0.001):
		self.model = model.to(device)
		self.device = device
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(model.parameters(), lr=lr)
		
		self.train_losses = []
		self.train_accuracies = []
	
	def train_epoch(self, train_loader):
		"""Train for one epoch"""
		self.model.train()
		epoch_loss = 0
		correct = 0
		total = 0
		
		for data, labels in train_loader:
			data, labels = data.to(self.device), labels.to(self.device)
			
			self.optimizer.zero_grad()
			outputs = self.model(data)
			loss = self.criterion(outputs, labels)
			loss.backward()
			self.optimizer.step()
			
			epoch_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		
		avg_loss = epoch_loss / len(train_loader)
		avg_accuracy = 100 * correct / total
		
		self.train_losses.append(avg_loss)
		self.train_accuracies.append(avg_accuracy)
		
		return avg_loss, avg_accuracy
	
	def evaluate(self, test_loader):
		"""Evaluate model on test set"""
		self.model.eval()
		all_preds = []
		all_labels = []
		
		with torch.no_grad():
			for data, labels in test_loader:
				data = data.to(self.device)
				outputs = self.model(data)
				_, predicted = torch.max(outputs.data, 1)
				
				all_preds.extend(predicted.cpu().numpy())
				all_labels.extend(labels.numpy())
		
		accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
		precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
		recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
		f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
		
		return accuracy, precision, recall, f1
	
	def plot_metrics(self, network_id=0, save_path='training_metrics.png'):
		"""Plot training metrics"""
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
		
		ax1.plot(self.train_losses, 'b-')
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')
		ax1.set_title(f'Network {network_id}: Training Loss')
		ax1.grid(True)
		
		ax2.plot(self.train_accuracies, 'g-')
		ax2.set_xlabel('Epoch')
		ax2.set_ylabel('Accuracy (%)')
		ax2.set_title(f'Network {network_id}: Training Accuracy')
		ax2.grid(True)
		
		plt.tight_layout()
		plt.savefig(save_path)
		print(f"Training metrics plot saved to {save_path}")
		plt.close()


def extract_features(model, data_loader, device='cuda'):
	"""
	Extract features from trained CNN for RayBNN
	Args:
		model: Trained CNN model
		data_loader: DataLoader for the dataset
		device: Device to use
	Returns:
		features (numpy array), labels (numpy array)
	"""
	model.to(device)
	model.eval()
	
	all_features = []
	all_labels = []
	
	with torch.no_grad():
		for data, labels in data_loader:
			data = data.to(device)
			# Get features from penultimate layer (256-dim)
			_, features = model(data, return_features=True)
			
			all_features.append(features.cpu())
			all_labels.append(labels)
	
	# Concatenate all batches
	all_features = torch.cat(all_features, dim=0).numpy()
	all_labels = torch.cat(all_labels, dim=0).numpy()
	
	return all_features, all_labels


def train_single_cnn(epochs=20, device='cuda', data_path='./mnist'):
	"""Train a single CNN (simpler approach like the working code)"""
	
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])
	
	trainset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
	testset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
	
	train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
	test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
	
	print(f"\n{'='*50}")
	print(f"Training Single CNN")
	print(f"{'='*50}")
	
	# Create model
	model = CNN(input_channels=1, num_classes=10, input_size=28)
	
	# Try to load existing weights
	weight_file = "model_single.pth"
	try:
		state_dict = torch.load(weight_file, map_location=device)
		model.load_state_dict(state_dict, strict=False)
		print(f"✓ Loaded weights from {weight_file}")
	except (FileNotFoundError, RuntimeError) as e:
		if isinstance(e, FileNotFoundError):
			print(f"No saved weights found, training from scratch")
		else:
			print(f"⚠ Weights incompatible with current architecture, training from scratch")
	
	trainer = CNNTrainer(model, device=device)
	
	# Train
	for epoch in range(epochs):
		loss, acc = trainer.train_epoch(train_loader)
		print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Accuracy: {acc:.2f}%')
	
	# Evaluate CNN
	test_acc, precision, recall, f1 = trainer.evaluate(test_loader)
	print(f'\n✓ CNN Training Complete!')
	print(f'  Test Accuracy: {test_acc:.2f}%')
	print(f'  Precision: {precision:.4f}')
	print(f'  Recall: {recall:.4f}')
	print(f'  F1 Score: {f1:.4f}')
	
	# Save model
	torch.save(model.state_dict(), weight_file)
	print(f'✓ Model saved to {weight_file}')
	
	# Plot metrics
	trainer.plot_metrics(network_id=1, save_path='training_metrics_cnn.png')
	
	return model, train_loader, test_loader





def plot_raybnn_metrics(train_loss, test_loss, train_acc, test_acc, save_path='raybnn_metrics.png'):
	"""Plot RayBNN training and test metrics"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
	
	epochs = range(1, len(train_loss) + 1)
	
	# Loss plot
	ax1.plot(epochs, train_loss, 'b-', label='Train Loss', marker='o')
	ax1.plot(epochs, test_loss, 'r-', label='Test Loss', marker='s')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax1.set_title('RayBNN: Training vs Test Loss')
	ax1.legend()
	ax1.grid(True)
	
	# Accuracy plot
	ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy', marker='o')
	ax2.plot(epochs, test_acc, 'r-', label='Test Accuracy', marker='s')
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Accuracy')
	ax2.set_title('RayBNN: Training vs Test Accuracy')
	ax2.legend()
	ax2.grid(True)
	
	plt.tight_layout()
	plt.savefig(save_path)
	print(f"\n✓ RayBNN metrics plot saved to {save_path}")
	plt.close()


def train_raybnn(x_train, y_train, x_test, y_test):
	"""
	Train RayBNN on extracted CNN features
	Args:
		x_train: Training features from CNN (shape: N x feature_dim)
		y_train: Training labels
		x_test: Test features from CNN
		y_test: Test labels
	Returns:
		output predictions
	"""
	# Auto-detect feature dimension
	input_size = x_train.shape[1]
	max_input_size = input_size
	#max_input_size = max(4096, input_size)
	
	print(f"\n{'='*50}")
	print("RayBNN Configuration:")
	print(f"{'='*50}")
	print(f"Feature dimension: {input_size}")
	print(f"Training samples: {x_train.shape[0]}")
	print(f"Test samples: {x_test.shape[0]}")
	
	# Track metrics for both train and test
	train_accuracy_values = []
	train_precision_values = []
	train_recall_values = []
	train_f1_values = []
	train_loss_values = []
	
	test_accuracy_values = []
	test_precision_values = []
	test_recall_values = []
	test_f1_values = []
	test_loss_values = []
	
	if isinstance(x_train, torch.Tensor):
		x_train = x_train.cpu().numpy()
	if isinstance(x_test, torch.Tensor):
		x_test = x_test.cpu().numpy()

	# Normalize features
	max_value = np.max(x_train)
	min_value = np.min(x_train)
	mean_value = np.mean(x_train)

	x_train = (x_train.astype(np.float32) - mean_value) / (max_value - min_value)
	x_test = (x_test.astype(np.float32) - mean_value) / (max_value - min_value)

	dir_path = "/tmp/"

	max_output_size = 10
	output_size = 10

	# Reduced parameters to avoid memory issues
	max_neuron_size = 1000
	batch_size = 100
	traj_size = 1
	proc_num = 2
	active_size = 500

	training_samples = x_train.shape[0]
	crossval_samples = x_train.shape[0]
	testing_samples = x_test.shape[0]

	# Format dataset for RayBNN
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
	print(f"\nCreating RayBNN architecture...")
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
		1000,  # Reduced from 10000
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

	max_epoch = 100
	stop_epoch = 100000
	stop_train_loss = 0.005
	max_alpha = 0.001
	exit_counter_threshold = 100000
	shuffle_counter_threshold = 200

	total_epochs = 10  # Reduced from 100

	print(f"\n{'='*50}")
	print("Training RayBNN")
	print(f"{'='*50}")

	for epoch in range(total_epochs):
		print(f"\nRayBNN Epoch {epoch + 1}/{total_epochs}...")
		max_epoch += 1
		
		try:
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

			# ===== EVALUATE ON TRAINING SET =====
			output_train = raybnn_python.test_network(train_x, arch_search)
			
			# Calculate train predictions and loss
			train_pred = [np.argmax(output_train[:, i % batch_size, 0, int(i/batch_size)]) 
						  for i in range(x_train.shape[0])]
			
			train_loss = 0.0
			for i in range(x_train.shape[0]):
				pred_probs = output_train[:, i % batch_size, 0, int(i/batch_size)]
				true_label = y_train[i]
				train_loss += -np.log(pred_probs[true_label] + 1e-10)
			train_loss /= x_train.shape[0]
			
			train_acc = accuracy_score(y_train, train_pred)
			train_ret = precision_recall_fscore_support(y_train, train_pred, average='macro')
			
			train_accuracy_values.append(train_acc)
			train_precision_values.append(train_ret[0])
			train_recall_values.append(train_ret[1])
			train_f1_values.append(train_ret[2])
			train_loss_values.append(train_loss)

			# ===== EVALUATE ON TEST SET =====
			test_x = np.zeros((input_size, batch_size, traj_size, testing_samples)).astype(np.float32)

			for i in range(x_test.shape[0]):
				j = (i % batch_size)
				k = int(i / batch_size)
				test_x[:, j, 0, k] = x_test[i, :]

			output_test = raybnn_python.test_network(test_x, arch_search)

			# Calculate test predictions and loss
			test_pred = [np.argmax(output_test[:, i % batch_size, 0, int(i/batch_size)]) 
						 for i in range(x_test.shape[0])]
			
			test_loss = 0.0
			for i in range(x_test.shape[0]):
				pred_probs = output_test[:, i % batch_size, 0, int(i/batch_size)]
				true_label = y_test[i]
				test_loss += -np.log(pred_probs[true_label] + 1e-10)
			test_loss /= x_test.shape[0]
			
			test_acc = accuracy_score(y_test, test_pred)
			test_ret = precision_recall_fscore_support(y_test, test_pred, average='macro')

			test_accuracy_values.append(test_acc)
			test_precision_values.append(test_ret[0])
			test_recall_values.append(test_ret[1])
			test_f1_values.append(test_ret[2])
			test_loss_values.append(test_loss)

			# Print both train and test metrics
			print(f'Epoch {epoch + 1}:')
			print(f'  Train - Loss: {train_loss:.5f}, Acc: {train_acc:.5f}, Prec: {train_ret[0]:.5f}, Rec: {train_ret[1]:.5f}, F1: {train_ret[2]:.5f}')
			print(f'  Test  - Loss: {test_loss:.5f}, Acc: {test_acc:.5f}, Prec: {test_ret[0]:.5f}, Rec: {test_ret[1]:.5f}, F1: {test_ret[2]:.5f}')		
		except Exception as e:
			print(f"❌ Error in epoch {epoch + 1}: {str(e)}")
			break

	# Print final results
	if test_accuracy_values:
		print(f"\n{'='*50}")
		print("RayBNN Training Complete")
		print(f"{'='*50}")
		print(f"Final Train - Loss: {train_loss_values[-1]:.5f}, Accuracy: {train_accuracy_values[-1]:.5f}")
		print(f"Final Test  - Loss: {test_loss_values[-1]:.5f}, Accuracy: {test_accuracy_values[-1]:.5f}")
		print(f"\nAverage Train - Acc: {np.mean(train_accuracy_values):.5f}, Prec: {np.mean(train_precision_values):.5f}, Rec: {np.mean(train_recall_values):.5f}, F1: {np.mean(train_f1_values):.5f}")
		print(f"Average Test  - Acc: {np.mean(test_accuracy_values):.5f}, Prec: {np.mean(test_precision_values):.5f}, Rec: {np.mean(test_recall_values):.5f}, F1: {np.mean(test_f1_values):.5f}")
		
		# Plot training curves
		plot_raybnn_metrics(train_loss_values, test_loss_values, 
						   train_accuracy_values, test_accuracy_values)
		
		return output_test.reshape(-1) if 'output_test' in locals() else None
	else:
		print("❌ No results produced - RayBNN training failed")
		return None

def main():
	"""
	Main pipeline: Single CNN → Feature Extraction → RayBNN
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}\n")
	
	print("="*60)
	print("STEP 1: Training Single CNN")
	print("="*60)
	
	# Step 1: Train single CNN (simpler approach)
	model, train_loader, test_loader = train_single_cnn(
		epochs=20,
		device=device,
		data_path='./mnist'
	)
	
	print("\n" + "="*60)
	print("STEP 2: Extracting Features for RayBNN")
	print("="*60)
	
	# Step 2: Extract features from the trained CNN
	# Create separate DataLoaders with larger batch sizes for faster feature extraction
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])
	
	trainset = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
	testset = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
	
	# Use much larger batch sizes for feature extraction (not training)
	train_loader_fast = DataLoader(trainset, batch_size=2000, shuffle=False, num_workers=4)
	test_loader_fast = DataLoader(testset, batch_size=2000, shuffle=False, num_workers=4)
	
	print("\nExtracting training features...")
	train_features, train_labels = extract_features(model, train_loader_fast, device)
	
	print("Extracting test features...")
	test_features, test_labels = extract_features(model, test_loader_fast, device)
	
	print(f'\n✓ Feature Extraction Complete!')
	print(f'  Train features shape: {train_features.shape}')
	print(f'  Test features shape: {test_features.shape}')
	print(f'  Feature dimension: {train_features.shape[1]}')
	
	# Save features
	np.save('mnist_train_features_cnn.npy', train_features)
	np.save('mnist_train_labels.npy', train_labels)
	np.save('mnist_test_features_cnn.npy', test_features)
	np.save('mnist_test_labels.npy', test_labels)
	print(f'✓ Features saved to .npy files')
	
	print("\n" + "="*60)
	print("STEP 3: Training RayBNN")
	print("="*60)
	
	# Step 4: Train RayBNN with extracted features
	try:
		output_y = train_raybnn(train_features, train_labels, test_features, test_labels)
		if output_y is not None:
			print("\n✓ Pipeline Complete!")
		else:
			print("\n⚠ RayBNN training had issues, but CNN features are saved")
	except MemoryError:
		print("\n❌ Out of Memory - Reduce batch_size or total_epochs in train_raybnn()")
	except KeyboardInterrupt:
		print("\n⚠ Training interrupted by user")
	except Exception as e:
		print(f"\n❌ RayBNN Error: {type(e).__name__}: {e}")
		print("CNN features are saved and can be used later")


if __name__ == '__main__':
	main()

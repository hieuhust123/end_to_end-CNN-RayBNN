import raybnn_python
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import pandas as pd
import matplotlib.pyplot as plt
import re
import sys


#from sklearn.datasets import load_iris as sklearn_load_iris
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.datasets import fetch_openml



def main():

	def debug(msg):
		"""Print to stderr so it always shows on terminal, even with > redirect."""
		sys.stderr.write(msg + "\n")
		sys.stderr.flush()

	## Load MNIST dataset

	#debug("[STEP 1] Loading MNIST dataset...")
	def load_mnist():
		X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
		#X=X.astype(np.float32) / 255.0
		x_train = X[:60000].reshape(-1, 28, 28)
		y_train = y[:60000].astype(int)
		x_test = X[60000:].reshape(-1, 28, 28)
		y_test = y[60000:].astype(int) 

		return x_train, y_train, x_test, y_test

	x_train, y_train, x_test, y_test = load_mnist()
	#debug("[STEP 1] Done loading MNIST.")

	# Normalize MNIST dataset
	max_value = np.max(x_train)
	min_value = np.min(x_train)
	mean_value = np.mean(x_train)

	x_train = (x_train.astype(np.float32) - mean_value)/(max_value - min_value)
	x_test = (x_test.astype(np.float32) - mean_value)/(max_value - min_value)

	# print("x_test after normalize", x_test)
	print("x_train shape:", x_train.shape)
	print("y_train shape:", y_train.shape)
	print("x_test shape:", x_test.shape)
	print("y_test shape:", y_test.shape)

	## Parameter setting for Fashion and MNIST dataset
	dir_path = "/tmp/"
	max_input_size = 784
	input_size = 784
	max_output_size = 10
	output_size = 10
	max_neuron_size = 2000
	batch_size = 1000
	traj_size = 1
	proc_num = 2
	active_size = 1000
	training_samples = 60
	crossval_samples = 60
	testing_samples = 10

	## Training params
	stop_strategy = "STOP_AT_TRAIN_LOSS"
	lr_strategy = "SHUFFLE_CONNECTIONS"
	lr_strategy2 = "MAX_ALPHA"
	loss_function = "sigmoid_cross_entropy_5"
	max_epoch = 200000
	stop_epoch = 1000000
	stop_train_loss = 0.005
	max_alpha = 0.01
	exit_counter_threshold = 200000
	shuffle_counter_threshold = 200

	# Device
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f"Using device: {device}")
	#debug(f"[STEP 1] Using device: {device}")

	class CaptureStdoutToFile:
		def __init__(self, filename):
			self.filename = filename
			self.stdout_fd = sys.stdout.fileno()
			self.saved_stdout_fd = os.dup(self.stdout_fd)
			self.f = None

		def __enter__(self):
			# Flush any buffered output
			sys.stdout.flush()
			# Open the log file
			self.saved_stdout_fd = os.dup(self.stdout_fd)
			self.f = open(self.filename, 'w')
			# Replace stdout with the log file
			os.dup2(self.f.fileno(), self.stdout_fd)
			return self

		def __exit__(self, exc_type, exc_value, traceback):
			# Flush and restore stdout
			sys.stdout.flush()
			os.dup2(self.saved_stdout_fd, self.stdout_fd)
			os.close(self.saved_stdout_fd)
			self.saved_stdout_fd = None
			if self.f:
				self.f.close()
				self.f = None

	class CNN_Feature_Extractor(nn.Module):
		def __init__(self, pretrained_path = None):
			super(CNN_Feature_Extractor, self).__init__()

			self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, 
			kernel_size=3, stride=1, padding=1)
			self.pool = nn.MaxPool2d(kernel_size=2)
			self.conv2 = nn.Conv2d(in_channels=12, out_channels=64, # out=12
			kernel_size=3, stride=1, padding=1)
			self.conv3 = nn.Conv2d(in_channels=64, out_channels=16, # out=24
			kernel_size=3, stride=1, padding=1)
			self.drop = nn.Dropout2d(p=0.1)

			if pretrained_path and os.path.isfile(pretrained_path):
				state_dict = torch.load(pretrained_path)
				self.load_state_dict(state_dict, strict=False)
				print(f"Loaded pretrained CNN features from {pretrained_path}")
			else:
				print("Using random CNN initialization")

		def forward(self, raw_images, verbose=False):
			# conv1 (28x28) -> pool (14x14)
			x = F.relu(self.pool(self.conv1(raw_images)))
			if verbose:
				print("After conv1 + pool + ReLU:", x.shape)
			
			# conv2 (14x14) -> pool (7x7)
			x = F.relu(self.pool(self.conv2(x)))
			if verbose:
				print("After conv2 + pool + ReLU:",x.shape)

			# conv3 (7x7) + dropout
			x = F.relu(self.drop(self.conv3(x)))
			if verbose:
				print("After conv3 + drop + ReLU:", x.shape)

			features_flat = x.reshape(x.size(0), -1)
			# if verbose:
			# 	debug(f"After flattening: {features_flat.shape}")
			# 	debug("\n" + "="*70)
			# 	debug("DIAGNOSTIC: CNN Features Output BEFORE go to RayBNN")
			# 	debug("="*70)
			# 	debug(f"  Mean: {features_flat.mean().item():.6f}")
			# 	debug(f"  Std: {features_flat.std().item():.6f}")
			# 	debug(f"  Min: {features_flat.min().item():.6f}")
			# 	debug(f"  Max: {features_flat.max().item():.6f}")
			# 	debug(f"  requires_grad: {features_flat.requires_grad}")
			return features_flat

	def extract_cnn_feature(model, data, batch_size=1000, device='cuda'
	, verbose = False):
		model.to(device)
		model.eval()
		features_list = []

		print(f"Processing {len(data)} samples in batches of {batch_size} on {device}")
		with torch.no_grad():
			for i in range(0, len(data), batch_size):
				batch = data[i:i+batch_size]
				batch_tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device)
				
				# Enable verbose only for first batch
				is_verbose = verbose and (i==0)
				features = model(batch_tensor, verbose = is_verbose)
				features_list.append(features.cpu().numpy())

				if (i // batch_size) % 10 == 0:
					print(f"  Processed {i + len(batch)}/{len(data)} samples")

		return np.vstack(features_list)

	def format_for_raybnn(features, labels, input_size, output_size, 
	batch_size, traj_size, num_samples):
		data_x = np.zeros((input_size, batch_size, traj_size, num_samples)).astype(np.float32)
		data_y = np.zeros((output_size, batch_size, traj_size, num_samples)).astype(np.float32)

		for i in range(min(features.shape[0], batch_size * num_samples)):
			j = i % batch_size
			k = int(i / batch_size)
		
			if k >= num_samples:
				break
				
			data_x[:, j, 0, k] = features[i, :]
			idx = int(labels[i])
			data_y[idx, j, 0, k] = 1.0
	
		return data_x, data_y

	def train_raybnn_model(train_x, train_y, crossval_x, crossval_y,
	arch_search, log_file="output_raybnn_training_with_cnn_features_784.txt"):
		print(f"Starting RayBNN training (output to {log_file})")
		with CaptureStdoutToFile(log_file):
			trained_model = raybnn_python.train_network(
				train_x, train_y, crossval_x, crossval_y,
				stop_strategy, lr_strategy, lr_strategy2,
				loss_function, max_epoch, stop_epoch, stop_train_loss,
				max_alpha, exit_counter_threshold, shuffle_counter_threshold,
				arch_search
			)
		return trained_model

	def evaluate_raybnn(model, test_x, test_y, y_test_labels, 
	batch_size ):
		output_y = raybnn_python.test_network(test_x, model)
		pred = []

		for i in range(len(y_test_labels)):
			j = i % batch_size
			k = int(i / batch_size)
			sample = output_y[:, j, 0, k]
			pred.append(np.argmax(sample))
	
		acc = accuracy_score(y_test_labels, pred)
		prec, rec, f1, _ = precision_recall_fscore_support(
			y_test_labels, pred, average='macro'
		)
		
		return {
			'accuracy': acc,
			'precision': prec,
			'recall': rec,
			'f1_score': f1,
			'predictions': pred
		} 

	def plot_training_loss(log_file, save_path):
		
		# Check if log file exists 
		if not os.path.exists(log_file):
			print(f"WARNING: Log file '{log_file}' not found. Skipping plot.")
			return
		
		try:
			loss_history = []
			with open(log_file, 'r') as f:
				for line in f:
					match = re.search(r"Train loss:\s*([\d\.]+)", line)
					if match:
						loss_history.append(float(match.group(1)))
		
		except IOError as e:
			print(f"ERROR: Could not read log file '{log_file}': {e}")
			return
		except Exception as e:
			print(f"ERROR: Unexpected error reading log file: {e}")
			return
		
		if loss_history:
			try:
				plt.figure(figsize=(10, 6))
				plt.plot(loss_history)
				plt.xlabel('Iteration')
				plt.ylabel('Loss')
				plt.title('RayBNN Training Loss with CNN Features')
				plt.grid(True)
				plt.savefig(save_path)
				plt.close()
				print(f"Plot saved to {save_path}")
			except Exception as e:
				print(f"ERROR: Could not save plot to '{save_path}': {e}")
		else:
			print("No loss history found in log file '{log_file}'")

	# Create RayBNN Architecture
	#debug("[STEP 2] Creating RayBNN Architecture...")
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
	# debug("[STEP 2] Done creating RayBNN Architecture.")

	# debug("[STEP 3] Creating CNN and extracting features...")
	cnn_model = CNN_Feature_Extractor(pretrained_path=None)
	
	# Extract features
	print("Extracting CNN features from training data...")
	train_features = extract_cnn_feature(cnn_model, x_train, 
	device=device, verbose=True)
	print(f"Training features shape: {train_features.shape}")
	sys.stdout.flush()
	
	diag_msg = (
		f"\n{'='*70}\n"
		f"DIAGNOSTIC: Extracted Training Features Analysis\n"
		f"{'='*70}\n"
		f"  Shape: {train_features.shape}\n"
		f"  Mean: {train_features.mean():.6f}\n"
		f"  Std: {train_features.std():.6f}\n"
		f"  Min: {train_features.min():.6f}\n"
		f"  Max: {train_features.max():.6f}\n"
		f"{'='*70}\n"
	)
	#print(diag_msg)
	#debug(diag_msg)  # Also print to stderr so it always shows

	print("Extracting CNN features from test data...")
	test_features = extract_cnn_feature(cnn_model, x_test, 
	device=device, verbose=False)
	print(f"Test features shape: {test_features.shape}")
	#debug("[STEP 3] Done extracting features.")
	
	# Format data
	train_x, train_y = format_for_raybnn(
		train_features, y_train, input_size, output_size,
		batch_size, traj_size, training_samples
	)
	# Use view instead of copy (saves memory)
	crossval_x, crossval_y = train_x, train_y
	test_x, test_y = format_for_raybnn(
		test_features, y_test, input_size, output_size,
		batch_size, traj_size, testing_samples
	)

	# Before training, add RayBNN input diagnostic
	# debug("[STEP 4] Formatting data for RayBNN...")
	# raybnn_diag = (
	# 	f"\n{'='*70}\n"
	# 	f"DIAGNOSTIC: RayBNN Input Data (train_x, train_y)\n"
	# 	f"{'='*70}\n"
	# 	f"  train_x shape: {train_x.shape}\n"
	# 	f"  train_y shape: {train_y.shape}\n"
	# 	f"  train_x mean: {train_x.mean():.6f}\n"
	# 	f"  train_x std: {train_x.std():.6f}\n"
	# 	f"  train_x min/max: [{train_x.min():.6f}, {train_x.max():.6f}]\n"
	# 	f"  train_y sum (samples per class): {train_y.sum(axis=(1,2,3))}\n"
	# 	f"{'='*70}\n"
	#)
	# print(raybnn_diag)
	# debug(raybnn_diag)  # Also print to stderr

	# Train RayBNN
	# NOTE: Use a DIFFERENT log file than the shell redirect target!
	log_file = "raybnn_training_log.txt"
	#debug("[STEP 5] Training RayBNN (this may take a while)...")
	trained_model = train_raybnn_model(
		train_x, train_y, crossval_x, crossval_y,
		arch_search, log_file
	)
	#debug("[STEP 5] Done training RayBNN.")
	
	#debug("[STEP 6] Evaluating...")
	test_labels = y_test[:testing_samples * batch_size].astype(int)
	print(f"Evaluating on {len(test_labels)} test samples")
	# Evaluate
	results = evaluate_raybnn(
		trained_model, test_x, test_y, test_labels, batch_size
	)

	results_msg = (
		f"\n{'='*50}\n"
		f"FINAL RESULTS:\n"
		f"{'='*50}\n"
		f"Accuracy:  {results['accuracy']:.4f}\n"
		f"Precision: {results['precision']:.4f}\n"
		f"Recall:    {results['recall']:.4f}\n"
		f"F1 Score:  {results['f1_score']:.4f}\n"
		f"{'='*50}\n"
	)
	#print(results_msg)
	#debug(results_msg)  # Also print to stderr

	# Plot
	#debug("[STEP 7] Plotting...")
	if os.path.exists(log_file):
		plot_training_loss(log_file, "training_loss_raybnn_with_cnn_features_plot.png")
	else:
		print(f"WARNING: Training log '{log_file}' was not created. Cannot plot training loss.")

	print("Done without errors!")
	#debug("[STEP 7] Done without errors!")

if __name__ == '__main__':
	main()
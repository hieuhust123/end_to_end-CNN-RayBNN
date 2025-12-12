import numpy as np
import raybnn_python
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import os 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris as sklearn_load_iris
from sklearn.model_selection import train_test_split


def main():


	def load_mnist():
		X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
		X = X.astype(np.float32) / 255.0
		y = y.astype(np.int64)

		x_train = X[:60000].reshape(-1, 28, 28)
		y_train = y[:60000]
		x_test = X[60000:].reshape(-1, 28, 28)
		y_test = y[60000:]

		return x_train, y_train, x_test, y_test

	x_train, y_train, x_test, y_test = load_mnist()

	#Normalize MNIST dataset
	max_value = np.max(x_train)
	min_value = np.min(x_train)
	mean_value = np.mean(x_train)

	x_train = (x_train.astype(np.float32) - mean_value)/(max_value - min_value)
	x_test = (x_test.astype(np.float32) - mean_value)/(max_value - min_value)




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


	#Format MNIST dataset
	train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
	train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)

	for i in range(x_train.shape[0]):
		j = (i % batch_size)
		k = int(i/batch_size)

		train_x[:, j , 0, k ] = x_train[i,:].flatten()

		idx = int(y_train[i])
		train_y[idx , j , 0, k ] = 1.0

	crossval_x = np.copy(train_x)
	crossval_y = np.copy(train_y)

	#Create Neural Network
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


	#Train Neural Network
	print("\n=== Starting Training ===")
	try:
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
		print("=== Training Completed Successfully ===\n")
	
	except Exception as e:
		print(f"ERROR during training: {e}")
		return
	if arch_search is None:
		print("ERROR: arch_search is None after training!")
		return
	
	print("=== Starting Test Data Preparation ===")
	test_x = np.zeros((input_size,batch_size,traj_size,testing_samples)).astype(np.float32)
	test_y = np.zeros((output_size,batch_size,traj_size,testing_samples)).astype(np.float32)
	print(f"test_x shape: {test_x.shape}")
	print(f"test_y shape: {test_y.shape}")
	print(f"x_test.shape[0]: {x_test.shape[0]}")
	for i in range(x_test.shape[0]):
		j = (i % batch_size)
		k = int(i/batch_size)

		if k >= testing_samples:
			print(f"WARNING: k={k} exceeds testing_samples={testing_samples}")
			break

		test_x[:, j , 0, k ] = x_test[i,:].flatten()
		idx = int(y_test[i])

		if idx < 0 or idx >= output_size:
			print(f"ERROR: Invalid label {idx} for sample {i}")
			continue

		test_y[idx , j , 0, k ] = 1.0
	
	print("=== Test data prepared. Starting inference ===\n")

	#Test Neural Network
	try:
		output_y, test_loss = raybnn_python.test_network(
			test_x,
			test_y,
			arch_search
		)
		print(f"Test Loss: {test_loss}")
	except Exception as e:
		print(f"ERROR during testing: {e}")
		import traceback
		traceback.print_exc()
		return

	if output_y is None:
		print("ERROR: output_y is None!")
		return
	pred = []
	for i in range(x_test.shape[0]):
		j = (i % batch_size)
		k = int(i/batch_size)

		if k >= output_y.shape[3]:
			print(f"WARNING: k={k} exceeds output_y batch dimension {output_y.shape[3]}")
			break
			
		sample = output_y[:, j , 0, k ]
		#print(sample)

		pred.append(np.argmax(sample))

	y_test = y_test.astype(int)  # Before passing to metrics

	acc = accuracy_score(y_test, np.array(pred).astype(int))

	ret = precision_recall_fscore_support(y_test, pred, average='macro')

	print("Accuracy: ",acc)
	print("Precision: ",ret[0])
	print("Recall: ",ret[1])
	print("F1 Score: ",ret[2])
	print("Support: ",ret[3])




if __name__ == '__main__':
	main()




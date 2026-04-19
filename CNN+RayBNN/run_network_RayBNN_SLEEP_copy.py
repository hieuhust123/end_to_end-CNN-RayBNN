import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import raybnn_python
from PIL import Image
import os
from torchvision import datasets, transforms,utils
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix


def kappa_metric(y_true, y_pred, n_cl = 4):
	# computes Cohen kappa per class
	y =  np.array(y_true) 
	y_ = np.array(y_pred) 
	res = []
	for c in range(n_cl):
		res.append(cohen_kappa_score(y==c, y_==c))
	return np.array(res)



def train_raybnn(x_train, y_train, x_test, y_test, training_samples, crossval_samples, testing_samples):
    kappa_values = []

    if isinstance(x_train, torch.Tensor):
        Rey_train = x_train.cpu().numpy()

    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value) / (max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value) / (max_value - min_value)

    print(x_train.shape)
    print(x_test.shape)


    dir_path = "/tmp/"

    max_input_size = 256
    input_size = 256

    max_output_size = 4
    output_size = 4

    max_neuron_size = 2000

    batch_size = 1000
    traj_size = 1

    proc_num = 2
    active_size = 1000

    # training_samples = 13752
    # crossval_samples = 13752
    # testing_samples = 4800

    # Format dataset
    train_x = np.zeros((input_size, batch_size, traj_size, training_samples )).astype(np.float32)  
    train_y = np.zeros((output_size, batch_size, traj_size, training_samples )).astype(np.float32)  

    for i in range(x_train.shape[0]):
        j = (i % batch_size)
        k = int(i // batch_size)
        train_x[:, j, 0, k] = x_train[i, :]
        train_y[:, j, 0, k] = y_train[i, :]
        # idx = int(y_train[i])
        # train_y[idx, j, 0, k] = 1.0

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

    total_epochs = 75

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

        test_x = np.zeros((input_size, batch_size, traj_size, testing_samples )).astype(np.float32)  
        test_y = np.zeros((output_size, batch_size, traj_size, testing_samples )).astype(np.float32)

        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i // batch_size)

            test_x[:, j, 0, k] = x_test[i, :]
            test_y[:, j, 0, k] = y_test[i, :]
            # print("y_test shape:",y_test.shape)

        # Test Neural Network
        output_y = raybnn_python.test_network(
            test_x,

            arch_search
        )

        print("output_y:", output_y.shape)

        pred = []
        y_label = []
        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i // batch_size)

            sample = output_y[:, j, 0, k]
            # print(sample)

            pred.append(np.argmax(sample))

        # pred = [np.argmax(output_y[:, i % batch_size, 0, int(i // batch_size)]) for i in range(x_test.shape[0])]
        pred = np.array(pred)
        y_label = [np.argmax(test_y[:, i % batch_size, 0, int(i // batch_size)]) for i in range(x_test.shape[0])]
        y_label = np.array(y_label)
        
        print("y_label:", y_label.shape, "pred.shape:", pred.shape)
        
        kappa_per = kappa_metric(y_label, pred, 4)
        
        kappa_overall = cohen_kappa_score(y_label, pred)
        
        print("Kappa coefficient per class:", kappa_per)
        print("Kappa value:", kappa_overall)

        kappa_values.append(kappa_overall)


    print(output_y.shape)
    return y_label, pred




if __name__ == '__main__':
    import os
    _script_start = time.time()
    HERE = os.path.dirname(os.path.abspath(__file__))
    predictions_dir = os.path.join(HERE, 'output', 'predictions')

    features_part1 = np.load(os.path.join(predictions_dir, 'features_part1_0.npy'))
    labels_part1 = np.load(os.path.join(predictions_dir, 'labels_part1_0.npy'))
    features_val_part1 = np.load(os.path.join(predictions_dir, 'features_val_part1_0.npy'))
    labels_val_part1 = np.load(os.path.join(predictions_dir, 'labels_val_part1_0.npy'))
    features_part2 = np.load(os.path.join(predictions_dir, 'features_part2_0.npy'))
    labels_part2 = np.load(os.path.join(predictions_dir, 'labels_part2_0.npy'))
    features_val_part2 = np.load(os.path.join(predictions_dir, 'features_val_part2_0.npy'))
    labels_val_part2 = np.load(os.path.join(predictions_dir, 'labels_val_part2_0.npy'))
    features_part3 = np.load(os.path.join(predictions_dir, 'features_part3_0.npy'))
    labels_part3 = np.load(os.path.join(predictions_dir, 'labels_part3_0.npy'))
    features_val_part3 = np.load(os.path.join(predictions_dir, 'features_val_part3_0.npy'))
    labels_val_part3 = np.load(os.path.join(predictions_dir, 'labels_val_part3_0.npy'))
    features_part4 = np.load(os.path.join(predictions_dir, 'features_part4_0.npy'))
    labels_part4 = np.load(os.path.join(predictions_dir, 'labels_part4_0.npy'))
    features_val_part4 = np.load(os.path.join(predictions_dir, 'features_val_part4_0.npy'))
    labels_val_part4 = np.load(os.path.join(predictions_dir, 'labels_val_part4_0.npy'))
    # print(outputs_CNN.shape)

    y_label_part1, pred_part1 = train_raybnn(features_part1, labels_part1, features_val_part1, labels_val_part1, int(len(features_part1)/1000), int(len(features_part1)/1000), int(len(features_val_part1)/1000))
    y_label_part2, pred_part2 = train_raybnn(features_part2, labels_part2, features_val_part2, labels_val_part2, int(len(features_part2)/1000), int(len(features_part2)/1000), int(len(features_val_part2)/1000))
    y_label_part3, pred_part3 = train_raybnn(features_part3, labels_part3, features_val_part3, labels_val_part3, int(len(features_part3)/1000), int(len(features_part3)/1000), int(len(features_val_part3)/1000))
    y_label_part4, pred_part4 = train_raybnn(features_part4, labels_part4, features_val_part4, labels_val_part4, int(len(features_part4)/1000), int(len(features_part4)/1000), int(len(features_val_part4)/1000))
    y_label = np.concatenate((y_label_part1, y_label_part2, y_label_part3, y_label_part4))
    pred = np.concatenate((pred_part1, pred_part2, pred_part3, pred_part4))

    kappa_per = kappa_metric(y_label, pred, 4)

    kappa_overall = cohen_kappa_score(y_label, pred)

    print("Kappa coefficient per class:", kappa_per)
    print("K value:", kappa_overall)

    _elapsed = time.time() - _script_start
    print(f"[run_network_RayBNN_SLEEP_copy.py] Total runtime: {_elapsed:.1f}s ({_elapsed/3600:.2f}h)")


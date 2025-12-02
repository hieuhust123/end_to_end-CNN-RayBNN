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

    ## Fashion-MNIST

    # def load_fashion_mnist():
    #     X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

    #     x_train = X[:60000].reshape(-1, 28, 28)
    #     y_train = y[:60000]
    #     x_test = X[60000:].reshape(-1, 28, 28)
    #     y_test = y[60000:]
    #     return x_train, y_train, x_test, y_test
    
    # x_train, y_train, x_test, y_test = load_fashion_mnist()


    ## MNIST
    def load_mnist():
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X=X.astype(np.float32) / 255.0

        x_train = X[:60000].reshape(-1, 28, 28)
        y_train = y[:60000]
        x_test = X[60000:].reshape(-1, 28, 28)
        y_test = y[60000:]



        return x_train, y_train, x_test, y_test

    x_train, y_train, x_test, y_test = load_mnist()


    ## IRIS
    # def load_iris():
    #     iris = sklearn_load_iris()
    #     X = iris.data
    #     y = iris.target

    #     # Split into train/test (70/30)
    #     x_train, x_test, y_train, y_test = train_test_split(X, y, 
    #     test_size=0.3, random_state=42, stratify=y
    # )
    #     # Reshape (add batch dims and 4th dims)
    #     # reshape to [features, samples, 1, 1]
    #     x_train = x_train.T.reshape(4, -1, 1, 1)
    #     x_test = x_test.T.reshape(4, -1, 1, 1)

    #     x_train = x_train.astype(np.float32)
    #     x_test = x_test.astype(np.float32)
    #     y_train = y_train.astype(np.float32)
    #     y_test = y_test.astype(np.float32)
        
    #     return x_train, y_train, x_test, y_test
    
    # x_train, y_train, x_test, y_test = load_iris()


    #Normalize MNIST dataset
    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value)/(max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value)/(max_value - min_value)

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    dir_path = "/tmp/"
    # Parameter setting Fashion-MNIST and MNIST

    max_input_size = 784
    input_size = 784

    max_output_size = 10
    output_size = 10

    max_neuron_size = 2000 #2000

    batch_size = 1000
    traj_size = 1

    proc_num = 2
    active_size = 1000 # 1000

    training_samples = 60
    crossval_samples = 60
    testing_samples = 10

    ## IRIS parameters setting

    # max_input_size = 4
    # input_size = 4

    # max_output_size = 3
    # output_size = 3

    # max_neuron_size = 2000

    # batch_size = 1000
    # traj_size = 1

    # proc_num = 2
    # active_size = 1000

    # training_samples = 105
    # crossval_samples = 45
    # testing_samples = 45


    #Format MNIST dataset
    train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
    train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)

    print("x_train shape: ",x_train.shape)
    print("train_x shape: ",train_x.shape)

    ## For Fashion-MNIST and MNIST dataset
    for i in range(x_train.shape[0]):
        j = (i % batch_size)
        k = int(i/batch_size)

        train_x[:, j , 0, k ] = x_train[i,:].flatten()

        idx = int(y_train[i])
        train_y[idx , j , 0, k ] = 1.0

    # ## For IRIS dataset
    # for i in range(x_train.shape[1]):
    #     j = (i % batch_size)
    #     k = int(i/batch_size)

    #     train_x[:, j , 0, k ] = x_train[:, i].flatten()

    #     idx = int(y_train[i])
    #     train_y[idx , j , 0, k ] = 1.0


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

    loss_function = "softmax_cross_entropy"

    max_epoch = 10000
    stop_epoch = 100000
    num_epochs = 100 # Set the total number of epochs you want to train for
    max_epoch = 1 # We will train one epoch at a time inside the loop
    stop_train_loss = 0.001

    max_alpha = 0.01

    exit_counter_threshold = 100000
    shuffle_counter_threshold = 200

    # print("Train X shape: ", train_x.shape)
    # print("Train Y shape: ", train_y.shape)
    # print("Cross val X: ", crossval_x)
    # print("Cross val Y: ", crossval_y)

    ######Train Neural Network
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

    test_x = np.zeros((input_size,batch_size,traj_size,testing_samples)).astype(np.float32)
    test_y = np.zeros((output_size,batch_size,traj_size,testing_samples)).astype(np.float32)
    
    ## Test dataset for Fashion-MNIST and MNIST
    for i in range(x_test.shape[0]):
        j = (i % batch_size)
        k = int(i/batch_size)

        test_x[:, j , 0, k ] = x_test[i, :].flatten()
        idx = int(y_test[i])
        test_y[idx, j, 0, k] = 1.0

    train_losses = []
    test_losses = []
    accuracies = []

    # ## For IRIS dataset
    # for i in range(x_test.shape[1]):
    #     j = (i % batch_size)
    #     k = int(i/batch_size)
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        
        ######Train Neural Network for one epoch
        arch_search = raybnn_python.train_network(
            train_x,
            train_y,
            crossval_x,
            crossval_y,
            stop_strategy,
            lr_strategy,
            lr_strategy2,
            loss_function,
            max_epoch, # This is set to 1
            epoch + 1, # Pass current epoch number
            stop_train_loss,
            max_alpha,
            exit_counter_threshold,
            shuffle_counter_threshold,
            arch_search
        )

        # Extract training loss from the returned architecture
        # NOTE: This assumes the rust code stores the last loss in the arch_search object.
        # You might need to adjust how you get the train_loss if it's handled differently.
        # For now, we'll parse it from the stdout.

    #     test_x[:, j , 0, k ] = x_test[:, i].flatten()
        #Test Neural Network
        output_y, test_loss_vec = raybnn_python.test_network(
            test_x,
            test_y,
            arch_search
        )

    #Test Neural Network
    output_y = raybnn_python.test_network(
        test_x,
        avg_test_loss = np.mean(test_loss_vec)
        test_losses.append(avg_test_loss)

        arch_search
    )
        # Calculate Accuracy
        pred = []
        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i/batch_size)
            sample = output_y[:, j , 0, k ]
            pred.append(np.argmax(sample))

    #print("Test Y shape: ",output_y.shape)
        acc = accuracy_score(y_test.astype(int), np.array(pred).astype(int))
        accuracies.append(acc)

    # Pred for Fashion-MNIST and MNIST dataset
    pred = []
    for i in range(x_test.shape[0]):
        j = (i % batch_size)
        k = int(i/batch_size)
        print(f"Epoch {epoch+1}: Test Loss = {avg_test_loss:.4f}, Accuracy = {acc:.4f}")

        sample = output_y[:, j , 0, k ]
        #print(sample)
    print("\n--- Final Results ---")
    # You can print final metrics or plot the results here
    # For example, plotting accuracy over epochs:
    plt.figure()
    plt.plot(range(1, num_epochs + 1), accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy over Epochs")
    plt.grid(True)
    plt.savefig("accuracy_over_epochs.png")
    plt.show()

        pred.append(np.argmax(sample))

    # # Pred for IRIS dataset
    # pred = []
    # for i in range(x_test.shape[1]):
    #     j = (i % batch_size)
    #     k = int(i/batch_size)

    #     sample = output_y[:, j , 0, k ]
    #     print(sample)

    #     pred.append(np.argmax(sample))

    y_test = y_test.astype(int)
    pred = np.array(pred).astype(int)
    # print("y_test types:", set(type(x) for x in y_test))

    # print("pred types:", set(type(x) for x in pred))

    acc = accuracy_score(y_test, pred)

    ret = precision_recall_fscore_support(y_test, pred, average='macro')

    print("Accuracy: ",acc)
    print("Precision: ",ret[0])
    print("Recall: ",ret[1])
    print("F1 Score: ",ret[2])
    print("Support: ",ret[3])
    # # plot a diagram
    # epochs = range(1,20)
    # plt.plot(epochs, acc)
    # plt.xlabel("Epochs")
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.ylabel("Accuracy")
    # plt.title("Training RayBNN with raw MNIST dataset")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("accuracy_curve_RayBNN_MNIST.png", dpi=300)
    # plt.show()



    print("Done without errors!")

if __name__ == '__main__':
    main()

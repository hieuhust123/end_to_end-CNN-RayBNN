import numpy as np
from urllib import request
import gzip
import pickle

# This is FASHION-MNIST dataset
filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def save_mnist():
    fashion_mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            fashion_mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            fashion_mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("fashion_mnist.pkl", 'wb') as f:
        pickle.dump(fashion_mnist,f)
    print("Save complete.")

def load():
    with open("fashion_mnist.pkl",'rb') as f:
        fashion_mnist = pickle.load(f)
    return fashion_mnist["training_images"], fashion_mnist["training_labels"], fashion_mnist["test_images"], fashion_mnist["test_labels"]

if __name__ == '__main__':
    save_mnist()
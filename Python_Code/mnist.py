import numpy as np
from urllib import request
import gzip
import pickle
import os

filename = [
["training_images","train-images.idx3-ubyte"],
["test_images","t10k-images.idx3-ubyte"],
["training_labels","train-labels.idx1-ubyte"],
["test_labels","t10k-labels.idx1-ubyte"]
]

# def download_mnist():
#     base_url = "http://yann.lecun.com/exdb/mnist/"
#     for name in filename:
#         print("Downloading "+name[1]+"...")
#         request.urlretrieve(base_url+name[1], name[1])
#     print("Download complete.")

def check_files_exist():
    missing_files = []
    for name in filename:
        if not os.path.exists(name[1]):
            missing_files.append(name[1])

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    if check_files_exist():
        save_mnist()
    else:
        print("MNIST dataset not found!, download it first or check file path")    

def load():

    if not os.path.exists("mnist.pkl"):
        print("mnist.pkl not found. Running init()...")
        init()
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()

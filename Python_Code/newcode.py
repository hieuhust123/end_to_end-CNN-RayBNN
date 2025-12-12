import numpy as np
import raybnn_python
import torch 
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		# First block
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.bn1 = nn. BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.pool1 = nn.MaxPool2d(2,2)





# transform = transforms.Compose([transforms.ToTensor(),
# 				transforms.Normalize((0.5,), (0.5, ))])
# trainset = torchvision.datasets.MNIST('mnist', train=True,
# 				download=True,
# 				transform=transform)
# testset = torchvision.datasets.MNIST('mnist', train=False,
# 				download=True, 
# 				transform=transform)
# train_loader = torch.utils.data.DataLoader(trainset, 
# 				batch_size = 100, 
# 				shuffle=True,
# 				num_workers=0)
# test_loader = torch.utils.data.DataLoader(testset,
# 				batch_size = 100, 
# 				shuffle = False, 
# 				num_workers = 0)


# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.features = nn.Sequential(
# 		nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
# 		nn.ReLU(inplace=True),
# 		nn.BatchNorm2d(32),
# 		nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
# 		nn.ReLU(inplace=True),
# 		nn.BatchNorm2d(32),
# 		nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
# 		nn.MaxPool2d(2,2),
# 		nn.ReLU(inplace=True),
# 		nn.BatchNorm2d(32),
# 		nn.Dropout(p=0.4),

# 		nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
# 		nn.ReLU(inplace=True),
# 		nn.BatchNorm2d(64),
# 		nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
# 		nn.ReLU(inplace=True),
# 		nn.BatchNorm2d(64),
# 		nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), 
# 		nn.MaxPool2d(2, 2),
# 		nn.ReLU(inplace=True),
# 		nn.BatchNorm2d(64),
# 		nn.Dropout(p=0.4)
# 		)

# 		self.classifier = nn.Sequential(
# 			nn.Linear(7 * 7 *64, 128),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(p=0.4),
# 			nn.Linear(128,10)
# 		)
# 	def forward(self, x, return_features=False):
# 		x = self.features(x)
# 		features = x.view(x.size(0), -1)
# 		if return_features:
# 			return features
# 		x = self.classifier(features)
# 		return x

# batch_size = 100
# epochs = 30

# num_networks = 6
# model_list = []
# for i in range(num_networks):
# 	model = Net()
# 	model_list.append(model)

# RandAffine = transforms.RandomAffine(degrees=10, translate = (0.1,0.1), scale = (0.8,1.2))
# transform = transforms.Compose([
# 	RandAffine,
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.5,), (0.5, )),
# ])

# for n_network in range(num_networks):
# 	model = model_list[n_network]
# 	criterion = nn.CrossEntropyLoss()
# 	optimizer = optim.Adam(model.parameters(), lr=0.001)
# 	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	device = torch.device("cpu")
# 	model.to(device)

# 	for epoch in range(epochs):
# 		trainset = torchvision.datasets.MNIST('mnist', train=True,
# 		download=True,
# 		transform=transform)

# 		train_loader = torch.utils.data.DataLoader(trainset, 
# 		batch_size=100,
# 		shuffle=True,
# 		num_workers=6)
# 		model.train()

# 		for i, data in enumerate(train_loader, 0):
# 			train_images, labels = data
# 			train_images, labels = train_images.to(device), labels.to(device)

# 			optimizer.zero_grad()
# 			outputs = model(train_images)
# 			loss = criterion(outputs, labels)
# 			loss.backward()
# 			optimizer.step()

# 	correct = 0
# 	total = 0
# 	transform1 = transforms.Compose([transforms.ToTensor(), 
# 				transforms.Normalize((0.5, ), (0.5, ))])
# 	trainset = torchvision.datasets.MNIST('mnist', train=True,
# 				download = True,
# 				transform=transform1)

# 	train_loader = torch.utils.data.DataLoader(trainset, 
# 		batch_size = 100,
# 		shuffle = True,
# 		num_workers = 6)
	
# 	model.eval()
# 	with torch.no_grad():
# 		for i, data in enumerate(train_loader, 0):
# 			train_images, labels = data
# 			train_images, labels = train_images.to(device), labels.to(device)
# 			train_features = model(train_images, return_features = True)
# 			train_features.append(features.cpu().numpy())
# 			train_labels.append(labels.numpy())
	
# 	train_features = np.concatenate(train_features, axis=0)
# 	train_labels = np.concatenate(train_labels, axis=0)

# 	print(f"Training features shape: {train_features.shape}")
# 	print(f"Training labels shape: {train_labels.shape}")


# transform = transforms.Compose([transforms.ToTensor(),
# 				transforms.Normalize((0.5, ), (0.5, ))])
# testset = torchvision.datasets.MNIST('mnist', train = False,
# 					download=True,
# 					transform=transform)
# test_loader = torch.utils.data.DataLoader(testset,
# 					batch_size = 10000,
# 					shuffle = False,
# 					num_workers=6)


# for n_network in range(num_networks):
# 	model = model_list[n_network]
# 	model.eval()
# 	with torch.no_grad():
# 		for i, data in enumerate(test_loader, 0):
# 			test_images, labels = data
# 			test_images, labels = test.to(device), labels.to(device)
# 			features = model(images, return_features=True)
# 			test_features_all.append(features.cpu().numpy())
# 			test_labels_all.append(labels.numpy())

# 	test_features = np.concatenate(test_features_all, axis=0)
# 	test_labels = np.concatenate(test_labels_all, axis=0)

# 	print(f"Test features shape: {test_features.shape}")
# 	print(f"Test labels shape: {test_labels.shape}")


import torch
torch.manual_seed(0) # for reproducibility
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import math
import numpy as np
from src.classification.readdata import Data
from src.os import path
from src.classification.network import MLP
from src.util.graphs import Graphs
from src.os import path

class CarClassification(object):
	"""docstring for CarClassification"""
	def __init__(self, arg):
		super(CarClassification, self).__init__()
		self.arg = arg
		self.graph = Graphs("Util loading.....")
		self.data = Data("loading...")
		self.device = torch.device(self._getDevice())
		self.model = MLP(6,[64,128,64],4).to(self.device)
		self.batch_size = 32
		self.learning_rate = 0.001
		self.epochs = 500
		self.loss = torch.nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
		self.path = path.ROOT_DIR

	def _getDevice(self):
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		if path.PLATFORM == "Darwin":
			device = 'mps'
		return device
	def runClassification(self):
		""" without using any mini-batches"""
		data = self.data.carDataset()
		X_train, X_test, y_train, y_test = self.data.splitDataTrainTest(data)

		# Convert numpy arrays to tensors
		x_train_tensors = torch.Tensor(X_train).float().to(self.device)
		x_test_tensors = torch.Tensor(X_test).float().to(self.device)
		y_train_tensors = torch.LongTensor(y_train).to(self.device)
		y_test_tensors = torch.LongTensor(y_test).to(self.device)

		# Builds train and test datasets
		train_dataset = TensorDataset(x_train_tensors, y_train_tensors)
		test_dataset = TensorDataset(x_test_tensors, y_test_tensors)

		# Builds a load of taking batch size
		train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size,shuffle=False)

		# train the model
		loss_values = [] # store the loss of each epoch
		for t in range(self.epochs):
			print(f'Epoch {t+1}\n.....................')
			loss_current_epochs = 0.0
			size = len(train_loader.dataset)
			for batch, (X, y) in enumerate(train_loader):
				# forward pass: Compute prediction and loss
				pred = self.model(X)
				loss = self.loss(pred,y)
				loss_current_epochs += loss.item()*X.size(0)
				# Backpropagation to update weight and make gradient zero before any weight update
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				if batch%self.batch_size == 0:
					loss, current = loss.item(), (batch+1)*len(X)
					print(f'loss: {loss:>7f} [{current: >5d}/{size: >5d}]')
			loss_values.append(loss_current_epochs/size)
		self.graph.drawLoss(loss_values)
		# save the models
		model_path = self.path+"models/car_evaluation.pth"
		torch.save(self.model.state_dict(),model_path)

		# evaluate the model performance
		correct = 0
		total_sample = 0 
		with torch.no_grad():
			for _, (X,y) in enumerate(test_loader):
				# get the prediction 
				pred = self.model(X)
				_, predicted_class = torch.max(pred.data,1)
				total_sample += y.size(0)
				correct += (predicted_class==y).sum().item()
		print(f'Accuracy of the network on the test size: {np.round(100*correct/total_sample,2)}%')


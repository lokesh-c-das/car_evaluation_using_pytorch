import torch
torch.manual_seed(0) # for reproducibility
import math
import torch.nn as nn
class MLP(nn.Module):
	"""docstring for MLP"""
	def __init__(self, input_size, hidden_size, num_classes):
		super(MLP, self).__init__()
		self.l1 = nn.Linear(input_size,hidden_size[0])
		self.ac1 = nn.ReLU()
		self.l2 = nn.Linear(hidden_size[0],hidden_size[1])
		self.ac2 = nn.ReLU()
		self.l3 = nn.Linear(hidden_size[1], hidden_size[2])
		self.ac3 = nn.ReLU()
		self.l4 = nn.Linear(hidden_size[2],num_classes)

	def forward(self, x):
		x = self.l1(x)
		x = self.ac1(x)
		x = self.l2(x)
		x = self.ac2(x)
		x = self.l3(x)
		x = self.ac3(x)
		x = self.l4(x)
		return x

		
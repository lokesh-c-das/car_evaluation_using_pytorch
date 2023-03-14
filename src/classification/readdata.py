import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import StandardScaler
from src.os import path
import torch
torch.manual_seed(0) # for reproducibility

class Data(object):
	"""docstring for Data"""
	def __init__(self, arg):
		super(Data, self).__init__()
		self.arg = arg
		self.path = path.ROOT_DIR
		print(f'Current Path: {self.path}')
	def splitDataTrainTest(self, data, test_size=0.1, random_state=123,shuffle=True):
		X, y = data
		X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_size,random_state=random_state,shuffle=shuffle)
		X_train, X_test = self.standarizeData(X_train, X_test)
		return X_train, X_test, np.array(y_train), np.array(y_test)

	def standarizeData(self, X_train, X_test):
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.fit_transform(X_test)
		return X_train, X_test
	def carDataset(self):
		data = pd.read_csv(self.path+"datasets/car/car.data")
		"""The attributes are strings by defaults, so we need to convert numerical values"""
		label_encoder = preprocessing.LabelEncoder()
		buying = label_encoder.fit_transform(list(data['buying']))
		maint = label_encoder.fit_transform(list(data['maint']))
		doors = label_encoder.fit_transform(list(data['doors']))
		persons = label_encoder.fit_transform(list(data['persons']))
		lug_boot = label_encoder.fit_transform(list(data['lug_boot']))
		safety = label_encoder.fit_transform(list(data['safety']))
		car_class = label_encoder.fit_transform(list(data['class']))
		"""Create CSV  file after conversion"""
		numerical_data = list(zip(buying, maint, doors, persons, lug_boot, safety, car_class))
		"""Creating features and target class"""
		X = list(zip(buying, maint, doors, persons, lug_boot, safety)) # Features
		Y = list(car_class) # Corresponding Target Class

		return X, Y #  return features and target




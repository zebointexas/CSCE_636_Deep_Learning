import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
	"""Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches
			are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072].
			(dtype=np.float32)
		y_train: An numpy array of shape [50000,].
			(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072].
			(dtype=np.float32)
		y_test: An numpy array of shape [10000,].
			(dtype=np.int32)
	"""

	### YOUR CODE HERE
	# Get training and testing filenames
	training_files = [
		os.path.join(data_dir, 'data_batch_%d' % i)
		for i in range(1, 6)
	]
	testing_file = os.path.join(data_dir, 'test_batch')

	# Load the training dataset
	x_train = []
	y_train = []
	for training_file in training_files:	
		with open(training_file, 'rb') as f:
			d = pickle.load(f, encoding='bytes')
		x_train.append(d[b'data'].astype(np.float32))
		y_train.append(np.array(d[b'labels'], dtype=np.int32))
	x_train = np.concatenate(x_train, axis=0)
	y_train = np.concatenate(y_train, axis=0)

	# Load the testing dataset
	with open(testing_file, 'rb') as f:
		d = pickle.load(f, encoding='bytes')
	x_test = d[b'data'].astype(np.float32)
	y_test = np.array(d[b'labels'], dtype=np.int32)
	### END CODE HERE

	return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=45000):
	"""Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
	x_train_new = x_train[:split_index]
	y_train_new = y_train[:split_index]
	x_valid = x_train[split_index:]
	y_valid = y_train[split_index:]

	return x_train_new, y_train_new, x_valid, y_valid


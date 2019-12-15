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

	negatives = False
# #
	meta_data_dict = unpickle(data_dir + "/batches.meta")
	cifar_label_names = meta_data_dict[b'label_names']
	cifar_label_names = np.array(cifar_label_names)

	# training data
	cifar_train_data = None
	cifar_train_labels = []

	# cifar_train_data_dict
	# 'batch_label': 'training batch 5 of 5'
	# 'data': ndarray
	# 'filenames': list
	# 'labels': list

	for i in range(1, 6):
		cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
		if i == 1:
			cifar_train_data = cifar_train_data_dict[b'data']
		else:
			cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
		cifar_train_labels += cifar_train_data_dict[b'labels']

	cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3072))

	cifar_train_labels = np.array(cifar_train_labels)

	cifar_test_data_dict = unpickle(data_dir + "/test_batch")
	cifar_test_data = cifar_test_data_dict[b'data']
	cifar_test_labels = cifar_test_data_dict[b'labels']

	cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3072))

	'''
	if negatives:
		cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
	else:
		cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
	'''

	cifar_test_labels = np.array(cifar_test_labels)

	return cifar_train_data.astype(np.float32), \
		   cifar_train_labels.astype(np.int32), \
		   cifar_test_data.astype(np.float32), \
		   cifar_test_labels.astype(np.int32), \

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


def unpickle(file):
    """load the cifar-10 data"""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data



############################################################
cifar_10_dir = 'cifar-10-batches-py'

train_data, train_labels, test_data, test_labels = load_data(cifar_10_dir)


print(train_data[1].shape)

print("X_Train: ", train_data.shape)
print("Y_Train: ", train_labels.shape)
print("X_Test: ", test_data.shape)
print("Y_Test: ", test_labels.shape)

print("************************************************************************************************************************")


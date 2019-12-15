import numpy as np
import random

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
	"""Parse a record to an image and perform data preprocessing.

	Args:
		record: An array of shape [3072,]. One row of the x_* matrix.
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	# Reshape from [depth * height * width] to [depth, height, width].
	# depth_major = tf.reshape(record, [3, 32, 32])
	depth_major = record.reshape((3, 32, 32))

	# Convert from [depth, height, width] to [height, width, depth]
	# image = tf.transpose(depth_major, [1, 2, 0])
	image = np.transpose(depth_major, [1, 2, 0])

	image = preprocess_image(image, training)

	return image

def preprocess_image(image, training):
	"""Preprocess a single image of shape [height, width, depth].

	Args:
		image: An array of shape [32, 32, 3].
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	if training:

		#---------##    YOUR CODE HERE    #---------##
		# Resize the image to add four extra pixels on each side.
		# image = tf.image.resize_image_with_crop_or_pad(image, 32 + 8, 32 + 8)
		image = np.pad(image, ((4, 4), (4, 4), (0, 0)), 'constant', constant_values=0)
		### END CODE HERE

		#---------##    YOUR CODE HERE    #---------##
		# Randomly crop a [32, 32] section of the image.
		# image = tf.random_crop(image, [32, 32, 3])
		# HINT: randomly generate the upper left point of the image
		a = random.randint(0, 8)
		b = random.randint(0, 8)
		image = image[a:a+32, b:b+32, :]
		### END CODE HERE ###

		#---------##    YOUR CODE HERE    #---------##
		# Randomly flip the image horizontally.
		# image = tf.image.random_flip_left_right(image)
		m = random.randint(0, 1)
		if m == 1:
		   image = image.fliplr(image)   # 圖片翻轉

		### END CODE HERE

		#---------## YOUR CODE HERE #---------##
		# Subtract off the mean and divide by the standard deviation of the pixels.
		# image = tf.image.per_image_standardization(image)
		# Reference here: https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
		mean = np.mean(image)
		standard_deviation = np.std(image)
		adjust = max(standard_deviation - 1.0/np.sqrt(3072))

		image = (image - mean)/adjust
		### END CODE HERE

	return image


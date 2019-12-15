import tensorflow as tf

"""This script defines the network.
"""

class ResNet(object):

	def __init__(self, resnet_version, resnet_size, num_classes, first_num_filters):
		"""Define hyperparameters.

		Args:
			resnet_version: 1 or 2. If 2, use the bottleneck blocks.
			resnet_size: A positive integer (n).
			num_classes: A positive integer. Define the number of classes.
			first_num_filters: An integer. The number of filters to use for the first block layer of the model.
			                   This number is then doubled for each subsampling block layer.  # &&& 什麼意思
		"""
		self.resnet_version = resnet_version
		self.resnet_size = resnet_size
		self.num_classes = num_classes
		self.first_num_filters = first_num_filters

	def __call__(self, inputs, training):
		"""Classify a batch of input images.

		Architecture (first_num_filters = 16):

		layer_name      | start | stack1 | stack2 | stack3 | output
		output_map_size | 32x32 | 32×32  | 16×16  | 8×8    | 1x1
		#layers 		| 1 	| 2n/3n	 | 2n/3n  | 2n/3n  | 1
		#filters 		| 16 	| 16(*4) | 32(*4) | 64(*4) | num_classes

		n = #residual_blocks in each stack layer = self.resnet_size
		The standard_block has 2 layers each.
		The bottleneck_block has 3 layers each.

		Example of replacing:
		standard_block 		conv3-16 + conv3-16
		bottleneck_block	conv1-16 + conv3-16 + conv1-64

		Args:
			inputs: A Tensor representing a batch of input images.
			training: A boolean. Used by operations that work differently in training and testing phases.

		Returns:
			A logits Tensor of shape [<batch_size>, self.num_classes].
		"""
		
		outputs = self._start_layer(inputs, training)

		if self.resnet_version == 1:
			block_fn = self._standard_block_v1
		else:
			block_fn = self._bottleneck_block_v2

		for i in range(3):
			filters = self.first_num_filters * (2**i)
			strides = 1 if i == 0 else 2
			outputs = self._stack_layer(outputs, filters, block_fn, strides, training)

		outputs = self._output_layer(outputs, training)

		return outputs

	################################################################################
	# Blocks building the network
	################################################################################
	def _batch_norm_relu(self, inputs, training):
		"""Perform batch normalization then relu."""

		### YOUR CODE HERE
		batch_norm_outputs = tf.layers.batch_normalization(inputs, training=training)
		outputs = tf.nn.relu(batch_norm_outputs)    # '''  這裡為何要relu   '''
	 	#Refer: https://www.tensorflow.org/api_docs/python/tf/compat/v1/layers/batch_normalization
		#Refer: https://www.tensorflow.org/api_docs/python/tf/nn/relu
		### END CODE HERE

		return outputs

	def _start_layer(self, inputs, training):
		"""Implement the start layer.

		Args:
			inputs: A Tensor of shape [<batch_size>, 32, 32, 3].
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: A Tensor of shape [<batch_size>, 32, 32, self.first_num_filters].
		"""
		
		### YOUR CODE HERE
		# initial conv1
		'''
		filters_i = tf.random.truncated_normal([3, 3, 3, self.first_num_filters])  # &&& 為何是這個形狀
												# 第1-2個，是kernel尺寸
												# 第3個，是kernel的channel
												# 第4個，kernel的數目
		           # truncated_normal:  https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal
		filters_ii = tf.Variable(filters_i)
				  # 把filters_i變成一個變量
		'''
		outputs_i = tf.layers.conv2d(inputs, filters = self.first_num_filters, kernel_size=3, strides=1, padding='same')

		### END CODE HERE

		# We do not include batch normalization or activation functions in V2
		# for the initial conv1 because the first block unit will perform these
		# for both the shortcut and non-shortcut paths as part of the first
		# block's projection.
		if self.resnet_version == 1:
			outputs = self._batch_norm_relu(outputs_i, training)

		return outputs

	def _output_layer(self, inputs, training):
		"""Implement the output layer.

		Args:
			inputs: A Tensor of shape [<batch_size>, 8, 8, channels].
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: A logits Tensor of shape [<batch_size>, self.num_classes].
		"""

		# Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
		if self.resnet_version == 2:
			inputs = self._batch_norm_relu(inputs, training)

		### YOUR CODE HERE
			inputs_i = tf.layers.AveragePooling2D(pool_size=2, padding="same", strides=1)(inputs)

			if self.resnet_version == 1:
				channels = 64
			else: channels = 4*64

			inputs_ii = tf.reshape(inputs_i, [-1, 8 * 8 * channels])

			outputs = tf.layers.dense(inputs_ii, 10)

		#        outputs = tf.nn.softmax(outputs, axis=-1)

		### END CODE HERE

		return outputs

	def _stack_layer(self, inputs, filters, block_fn, strides, training):
		"""Creates one stack of standard blocks or bottleneck blocks.
		
		Args:
			inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
			filters: A positive integer. The number of filters for the first
				convolution in a block.
			block_fn: 'standard_block' or 'bottleneck_block'.
			strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: The output tensor of the block layer.
		"""

		filters_out = filters * 4 if self.resnet_version == 2 else filters

		def projection_shortcut(inputs):
			### YOUR CODE HERE

			output_1_x_1 = tf.layers.conv2d(inputs, filters=filters_out, kernel_size=1, strides=strides, padding='valid')
			output = tf.layers.batch_normalization(output_1_x_1, training=training)

			print(" -----  now begin projection_shortcut  -----")
			print(" shape = " + str(output.shape))
			return output


			### END CODE HERE

		### YOUR CODE HERE
		# Only the first block per stack_layer uses projection_shortcut

		for k in range(self.resnet_size):

			if self.resnet_version == 1:
				print("with resnet_version " + str(self.resnet_version) )
				if k == 0 and strides == 2:
					print("with k == " + str(k) + " strides = " + str(strides) )
					outputs = block_fn(inputs, filters_out, training, projection_shortcut, strides)
				else:
					print("with k == " + str(k) + " strides = " + str(strides) )
					outputs = block_fn(inputs, filters_out, training, None, strides)

			if self.resnet_version == 2:
				print("with resnet_version " + str(self.resnet_version))
				if k == 0:
					outputs = block_fn(inputs, filters_out, training, projection_shortcut, strides)
				else:
					outputs = block_fn(inputs, filters_out, training, None, 1)

		### END CODE HERE

		return outputs

	def _standard_block_v1(self, inputs, filters, training, projection_shortcut, strides):
		"""Creates a standard residual block for ResNet v1.
		
		Args:
			inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
			filters: A positive integer. The number of filters for the first
				convolution.
			training: A boolean. Used by operations that work differently
				in training and testing phases.
			projection_shortcut: The function to use for projection shortcuts
      			(typically a 1x1 convolution when downsampling the input).
			strides: A positive integer. The stride to use for the block. If
				greater than 1, this block will ultimately downsample the input.

		Returns:
			outputs: The output tensor of the block layer.
		"""

		print("*******  Now begin standard residual block  *******")
		s = inputs

		if projection_shortcut is None:
			print("&&&&& projection_shortcut is None, then shortcut size = " + str(s.shape))

		if projection_shortcut is not None:
			### YOUR CODE HERE
			s = projection_shortcut(inputs)
			### END CODE HERE

		### YOUR CODE HERE

		f_i_1 = tf.random.truncated_normal( [3, 3, filters, filters] )
		f_i_2 = tf.random.truncated_normal( [3, 3, filters, filters] )

		f_1 = tf.Variable(f_i_1)
		f_2 = tf.Variable(f_i_2)

		# need "shortcut" for "outputs2"

		print("2&&&&& projection_shortcut is None, then shortcut size = " + str(s.shape))

		if strides == 1:
			o_1 = tf.layers.conv2d(s, filters=filters, padding='same', kernel_size=3, strides=1)

			print("o_1's shape is " + str(o_1.shape))
			o_1 = self._batch_norm_relu(o_1, training)

			# for "outputs2"
			o_2 = tf.layers.conv2d(o_1, filters=filters, padding='same', kernel_size=3, strides=1)
			o_2_bn = tf.layers.batch_normalization(o_2, training=training)

		if strides == 2:
			o_1 = tf.layers.conv2d(s, filters=filters, padding='same', kernel_size=3, strides=1)

			print("o_1's shape is " + str(o_1.shape))
			o_1 = self._batch_norm_relu(o_1, training)
			print("o_1's shape is " + str(o_1.shape))

			# for "outputs2"
			o_2 = tf.layers.conv2d(o_1, filters=filters, padding='same', kernel_size=3, strides=1)
			print("o_2's shape is " + str(o_2.shape))

			o_2_bn = tf.layers.batch_normalization(o_2, training=training)
			print("o_2_bn's shape is " + str(o_2_bn.shape))

			'''
			o_1's shape is (?, 16, 16, 32)
			o_1's shape is (?, 16, 16, 32)
			o_2's shape is (?, 16, 16, 16)
			o_2_bn's shape is (?, 16, 16, 16)
			o_2_bn shape = (?, 16, 16, 16)
			shortcut shape = (?, 16, 16, 32)
			'''


		print( "o_2_bn shape = " + str(o_2_bn.shape) )
		print( "shortcut shape = " + str(s.shape) )

		o_2_bn += s



		output_final = tf.nn.relu(o_2_bn)

		### END CODE HERE

		return output_final

	def _bottleneck_block_v2(self, inputs, filters, training, projection_shortcut, strides):

		"""Creates a bottleneck block for ResNet v2.
		
		Args:
			inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
			filters: A positive integer. The number of filters for the first convolution. NOTE: filters_out will be 4xfilters.
			training: A boolean. Used by operations that work differently in training and testing phases.
			projection_shortcut: The function to use for projection shortcuts
      			(typically a 1x1 convolution when downsampling the input).

			strides: A positive integer. The stride to use for the block. If greater than 1, this block will ultimately downsample the input.

		Returns:
			outputs: The output tensor of the block layer.
		"""

		### YOUR CODE HERE

		print("*******  Now begin bottleneck block  *******")
		s = inputs

		if projection_shortcut is not None:
			s = projection_shortcut(inputs)

		# find channels
		in_channels = inputs.get_shape().as_list()[3]

        # 設置層次
		f_i_1 = tf.random.truncated_normal(  [1, 1,     in_channels,          int(filters / 4)  ]   )
		f_i_2 = tf.random.truncated_normal(  [3, 3,     int(  filters/4  ),   int(filters / 4)  ]   )
		f_i_3 = tf.random.truncated_normal(  [1, 1,     int(  filters/4  ),   filters   ]           )

		f_1 = tf.Variable(f_i_1)
		f_2 = tf.Variable(f_i_2)
		f_3 = tf.Variable(f_i_3)

		inputs = self._batch_norm_relu(inputs, training)

		o_1 = tf.layers.conv2d(inputs, filters=int(filters / 4), padding='same', kernel_size=1, strides=1)
		o_1 = self._batch_norm_relu(o_1, training)

		o_2 = tf.layers.conv2d(o_1, filters=int(filters / 4), padding='same', kernel_size=3, strides=strides)
		o_2 = self._batch_norm_relu(o_2, training)

		o_3 = tf.layers.conv2d(o_2, filters=filters, padding='same', kernel_size=1, strides=1)

		o_3_bn = tf.layers.batch_normalization(o_3, training=training)

		o_3_bn += s

		outputs = tf.nn.relu(o_3_bn)
		### END CODE HERE

		return outputs


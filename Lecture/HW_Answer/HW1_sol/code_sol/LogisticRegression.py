import numpy as np
import math
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
	def __init__(self, learning_rate, max_iter):
		self.learning_rate = learning_rate
		self.max_iter = max_iter

	def fit_GD(self, X, y):
		"""Train perceptron model on data (X,y) with GD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
		_, d = X.shape
		self.W = np.zeros(d)
		for i in range(self.max_iter):
			gradient = np.zeros(d)
			for j in range(len(X)):
				gradient_j = self._gradient(X[j], y[j])
				gradient += gradient_j
			gradient = np.dot(1/len(X), gradient)
			direction = np.dot(-1, gradient)
			self.W += self.learning_rate*direction
		### END YOUR CODE

		return self

	def fit_BGD(self, X, y, batch_size):
		"""Train perceptron model on data (X,y) with BGD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.
			batch_size: An integer.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
		_, d = X.shape
		self.W = np.zeros(d)
		for i in range(self.max_iter):
			gradient = np.zeros(d)
			J_sub = np.random.choice(len(X), batch_size, replace=False)
			for j in range(batch_size):
				gradient_j = self._gradient(X[J_sub[j]], y[J_sub[j]])
				gradient += gradient_j
			gradient = np.dot(1/batch_size, gradient)
			direction = np.dot(-1, gradient)
			self.W += self.learning_rate*direction

		### END YOUR CODE

		return self
		
	def fit_SGD(self, X, y):
		"""Train perceptron model on data (X,y) with SGD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
		_, d = X.shape
		self.W = np.zeros(d)
		for i in range(self.max_iter):
			m = np.random.randint(0, len(X))
			gradient = self._gradient(X[m], y[m])
			direction = np.dot(-1, gradient)
			self.W += self.learning_rate*direction
		### END YOUR CODE
		
		return self

	def _gradient(self, _x, _y):
		"""Compute the gradient of cross-entropy with respect to self.W
		for one training sample (_x, _y). This function is used in SGD.

		Args:
			_x: An array of shape [n_features,].
			_y: An integer. 1 or -1.

		Returns:
			_g: An array of shape [n_features,]. The gradient of
				cross-entropy with respect to self.W.
		"""
		### YOUR CODE HERE
		single_grad = - (_y * _x) / (1+math.exp(_y*np.dot(np.transpose(self.W), _x)))
		return single_grad

		### END YOUR CODE


	def get_params(self):
		"""Get parameters for this perceptron model.

		Returns:
			W: An array of shape [n_features,].
		"""
		if self.W is None:
			print("Run fit first!")
			sys.exit(-1)
		return self.W

	def predict_proba(self, X):
		"""Predict class probabilities for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds_proba: An array of shape [n_samples, 2].
				Only contains floats between [0,1].
		"""
		### YOUR CODE HERE
		n, _ = X.shape
		prob = []
		for i in range(n):
			pos_prob = 1 / (1 + math.exp(-np.dot(np.transpose(self.W), X[i])))
			prob.append([pos_prob, 1-pos_prob])
		return y

		### END YOUR CODE

	def predict(self, X):
		"""Predict class labels for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds: An array of shape [n_samples,]. Only contains 1 or -1.
		"""
		### YOUR CODE HERE
		n, _ = X.shape
		y = []
		for i in range(n):
			y_pred = 1 if np.dot(np.transpose(self.W), X[i]) >= 0  else -1
			y.append(y_pred)
		return y

		### END YOUR CODE

	def score(self, X, y):
		"""Returns the mean accuracy on the given test data and labels.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			score: An float. Mean accuracy of self.predict(X) wrt. y.
		"""
		### YOUR CODE HERE
		n, _ = X.shape
		corrects = 0
		for i in range(n):
			if y[i]*np.dot(np.transpose(self.W), X[i])>=0:
				corrects += 1
		return corrects/n

		### END YOUR CODE



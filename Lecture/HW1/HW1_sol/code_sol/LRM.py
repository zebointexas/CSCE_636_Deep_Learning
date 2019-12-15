#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_BGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        _, d = X.shape

        y = np.eye(self.k)[labels.astype(np.int32)]

        self.W = np.zeros([d, self.k])

        for i in range(self.max_iter):

            gradient = np.zeros([d, self.k])

            J_sub = np.random.choice(len(X), batch_size, replace=False)

            for j in range(batch_size):
                gradient_j = self._gradient(X[J_sub[j]], y[J_sub[j]])
                gradient += gradient_j

            gradient = np.dot(1/batch_size, gradient)
            direction = np.dot(-1, gradient)
            self.W += self.learning_rate*direction

		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector of shape [self.k,].

        Returns:
            _g: An array of shape [n_features, self.k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        d, = _x.shape
        x_reshape = np.reshape(_x, (1, d))

        logits = np.dot(x_reshape, self.W)
        
        k, = _y.shape
        
        probs = self.softmax(logits)

        sub_grad = (probs - _y)

        sub_grad_reshape = np.reshape(sub_grad, (1, k))
        
        single_grad =  np.dot(x_reshape.T, sub_grad_reshape)

        return single_grad

		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        exp_x = np.exp(x)

        probs = exp_x / np.sum(exp_x)

        return probs
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


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        z = np.dot(X, self.W)

        print('ZeboobbzZZZZZZZZZZZZZZZZZZZZZZZ')
        print(z.shape)
        
        preds = np.argmax(z, axis=1)

        return preds
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE

        print(preds.shape)    

        is_correct = (preds == labels).astype(np.float64)
        score = np.sum(is_correct) / is_correct.size

        return score
		### END YOUR CODE


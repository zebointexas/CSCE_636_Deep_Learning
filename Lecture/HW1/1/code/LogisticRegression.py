import numpy as np
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
        
        n_samples, n_features = X.shape
        
		### YOUR CODE HERE
        for i in range(400):
            gradient_sum = 0
            
            for t in range(len(X)-1): 
                gradient_sum = gradient_sum + self._gradient(X[t],y[t])
            
            gradient_aver = gradient_sum/len(X)
                
            v_t = -1 * gradient_aver
            
            self.W = self.W + self.learning_rate * v_t
        
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

        t = len(X) 
        
        batch_count = t/batch_size
         
        final_count = t - batch_count*batch_size
        
        i = 0
        
        while i < batch_count-1: 
      
            gradient_sum = 0
            
            for t in range(batch_size): 
                    gradient_sum = gradient_sum + self._gradient(X[i*batch_size+t],y[i*batch_size+t])
              
            gradient_aver = gradient_sum/batch_size
                
            v_t = -1 * gradient_aver
            
            self.W = self.W + self.learning_rate * v_t
             
            i = i+1
  
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
 
        
        for i in range(400):
            
            s = np.random.randint(0,len(X)-1)
            
            gradient = self._gradient(X[s],y[s])
       
            v_t = -1 * gradient
            
            self.W = self.W + self.learning_rate * v_t
        
		### END YOUR CODE



		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE

        _g =   ( (-1)*_y*_x )   / ( np.exp(   _y*(self.W).dot(_x)   ) + 1 ) 
            
        return _g
    
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
        p = lambda x : 1 / ( 1+ np.exp(-1*x) )
 
        p_binary = p(self.W.dot(X.T))
        
        sum = 0
        
        for i in p_binary: 
            sum = sum + i
        
        print("Probability for 1")
        
        return sum/len(p_binary)
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE

        print('predict --------')
        p = lambda x : 1 / ( 1+ np.exp(-1*x) )
 
        p_binary = p(self.W.dot(X.T))
        
        j = 0
        
        for i in p_binary: 
            if i > 0.5: 
                p_binary[j] = 1
            else:
                p_binary[j] = -1
                
            j = j + 1

      #  print( len(p_binary[p_binary==-1]) ) 

        return p_binary
    
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

        p = self.predict(X)
        
        correct_count = 0 
        
        j= 0 
        
        for i in p: 
            if i == y[j]: 
                correct_count = correct_count + 1
            j=j+1
                

        return correct_count/len(y)

		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self


    def test(self, weights):
        print(weights)
 
import numpy as np
from LogisticRegression import logistic_regression 

a = logistic_regression(learning_rate=0.5, max_iter=100)

x = np.array([5,7,8,1,5])
y = 5
w = np.array([1,1,1])
 

a.test('safsfdsggggg')

aa = a.assign_weights(w)
 
print(aa)


zzz = a._gradient(x,y)

print(zzz)

# logisticR_classifier._gradient(x,y)


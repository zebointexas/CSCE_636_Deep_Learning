#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:43:09 2019

@author: dior
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 

# Present the training and testing data points
##################################################
x_train = np.array([ [0],[2],[3],[5] ])                   
y_train = [1,4,9,16] 

x_test = np.array([ [1],[4] ])     
y_test = [3,12]

area = np.pi*30

plt.figure(figsize=(16,7))
plt.scatter(x_train, y_train, s=area, c='blue', alpha=1)
plt.scatter(x_test, y_test, s=area, c='red', alpha=1)
plt.title('Poly with 4-d')
plt.xlabel('x')
plt.ylabel('y')
##################################################

# LinearRegression
##################################################
# lin = LinearRegression() 
# lin.fit(x_train,y_train) 

# plt.plot(x_train, lin.predict(x_train), color = 'red') 
################################################## 
  

# D = 0 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 0) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin0 = LinearRegression() 
lin0.fit(X_poly,y_train) 
plt.plot(x_train, lin0.predict(poly.fit_transform(x_train)), color = 'purple', label = 'p = 0') 
train_result0 = lin0.predict(poly.fit_transform(x_train))  
test_result0 = lin0.predict(poly.fit_transform(x_test))
    
# D = 1 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 1) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin1 = LinearRegression() 
lin1.fit(X_poly,y_train) 
plt.plot(x_train, lin1.predict(poly.fit_transform(x_train)), color = 'green', label = 'p = 1')    
train_result1 = lin1.predict(poly.fit_transform(x_train))   
test_result1 = lin1.predict(poly.fit_transform(x_test))

# D = 2 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin2 = LinearRegression() 
lin2.fit(X_poly,y_train) 
plt.plot(x_train, lin2.predict(poly.fit_transform(x_train)), color = 'grey', label = 'p =2')     
train_result2 = lin2.predict(poly.fit_transform(x_train))  
test_result2 = lin2.predict(poly.fit_transform(x_test))   
    
# D = 3 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin3 = LinearRegression() 
lin3.fit(X_poly,y_train) 
plt.plot(x_train, lin3.predict(poly.fit_transform(x_train)), color = 'orange', label = 'p = 3')
train_result3 = lin3.predict(poly.fit_transform(x_train))       
test_result3 = lin3.predict(poly.fit_transform(x_test)) 
    
# D = 4 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin4 = LinearRegression() 
lin4.fit(X_poly,y_train) 
plt.plot(x_train, lin4.predict(poly.fit_transform(x_train)), color = 'blue', label = 'p =4')     
train_result4 = lin4.predict(poly.fit_transform(x_train))  
test_result4 = lin4.predict(poly.fit_transform(x_test))

# D = 8 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 8) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin20 = LinearRegression() 
lin20.fit(X_poly,y_train) 
      
test_result20 = lin20.predict(poly.fit_transform(x_test))


plt.legend()    
plt.show() 



# Training Result
################################################
 
train_result = ['','','','']
train_result = np.row_stack((train_result,train_result0)) 
train_result = np.row_stack((train_result,train_result1)) 
train_result = np.row_stack((train_result,train_result2)) 
train_result = np.row_stack((train_result,train_result3)) 
train_result = np.row_stack((train_result,train_result4)) 

 
train_result = np.delete(train_result, (0), axis=0)

print("------ train_result ------")
print(train_result)


# Test Result
################################################
 
# x_train_matrix = [0,0,0]
# x_train_matrix = np.row_stack((x_train_matrix,x_1)) 
# x_train_matrix = np.row_stack((x_train_matrix,x_2)) 
# x_train_matrix = np.row_stack((x_train_matrix,x_3)) 
# x_train_matrix = np.row_stack((x_train_matrix,x_4)) 
# x_train_matrix = np.delete(x_train_matrix, (0), axis=0)

test_result = ['','']
test_result = np.row_stack((test_result,test_result0)) 
test_result = np.row_stack((test_result,test_result1)) 
test_result = np.row_stack((test_result,test_result2)) 
test_result = np.row_stack((test_result,test_result3)) 
test_result = np.row_stack((test_result,test_result4)) 
test_result = np.delete(test_result, (0), axis=0)

print("------ test_result ------")
print(test_result)



# Typical Bias
################################################
x_test = np.array([ [1],[4] ])     
y_test = [3,12]


Typical_Bias_0 = pow( (float(test_result[0][0]) - float(x_test[0])),2 ) + pow( (float(test_result[0][1]) - float(x_test[1])), 2)/2
Typical_Bias_1 = pow( (float(test_result[1][0]) - float(x_test[0])),2 ) + pow( (float(test_result[1][1]) - float(x_test[1])), 2)/2
Typical_Bias_2 = pow( (float(test_result[2][0]) - float(x_test[0])),2 ) + pow( (float(test_result[2][1]) - float(x_test[1])), 2)/2
Typical_Bias_3 = pow( (float(test_result[3][0]) - float(x_test[0])),2 ) + pow( (float(test_result[3][1]) - float(x_test[1])), 2)/2
Typical_Bias_4 = pow( (float(test_result[4][0]) - float(x_test[0])),2 ) + pow( (float(test_result[4][1]) - float(x_test[1])), 2)/2

 
Typical_Bias = []
Typical_Bias.append(Typical_Bias_0)
Typical_Bias.append(Typical_Bias_1)
Typical_Bias.append(Typical_Bias_2)
Typical_Bias.append(Typical_Bias_3)
Typical_Bias.append(Typical_Bias_4)

print(Typical_Bias)
 
# Variance
################################################
 
Variance = []
k = 0 

print(len(Typical_Bias))

for i in range(len(Typical_Bias)): 
    Variance.append( Typical_Bias[k]/2 )
    k += 1
 
print(Variance)

# Total Error
################################################
Total_Error = [] 

for i in range(len(test_result)): 
    Total_Error.append(float(test_result[i][0]) - float(x_test[0])  +  float(test_result[i][1]) - float(x_test[1]) )

print(Total_Error)

# Training Error
################################################
x_train = np.array([ [0],[2],[3],[5] ])                   
y_train = [1,4,9,16] 

training_error = []
for i in train_result:# Get every training result
    
    k = 0 
    training_error_sum = 0
    for j in i:  
        training_error_sum = training_error_sum + pow( y_train[int(k)] - float(i[k]), 2 )       
        k += 1
    
    training_error.append(training_error_sum/4)

print("------- Training Error ------")
print(training_error)
     
 

# Test Error
################################################
testing_error = []

for i in test_result:# Get every training result
    
    k = 0 
    testing_error_sum = 0
    for j in i:  
        testing_error_sum = testing_error_sum + pow( y_test[int(k)] - float(i[k]), 2 )       
        k += 1
    
    testing_error.append(testing_error_sum/4)

print("------- Testing Error ------")
print(testing_error)
     

# Plot
################################################
x_point = [0,1,2,3,4]

plt.figure(figsize=(14,7))

plt.plot(x_point, Typical_Bias , color = 'yellow', label = 'Typical_Bias')
plt.plot(x_point, Variance , color = 'blue', label = 'Variance')
plt.plot(x_point, Total_Error , color = 'pink', label = 'Total_Error')
plt.plot(x_point, training_error , color = 'black', label = 'Training_error')
plt.plot(x_point, testing_error , color = 'grey', label = 'Testing_error')     
  
plt.title('Error')
plt.xlabel('x')
plt.ylabel('y')

plt.legend()    
plt.show() 

######################################################
# (3) Explain why each of the five curves has the shape displayed in part (2).












from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

random_seed = 1

x = df[['Lag1','Lag2']]
y = df['Direction']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = random_seed)
# train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train_data和test_dat

accuracy = []
for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    accuracy.append(np.mean(pred_i != y_test))

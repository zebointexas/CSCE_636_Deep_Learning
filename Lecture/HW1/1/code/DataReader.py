import numpy as np

"""This script implements the functions for reading data.
"""



######################################## load the data file
def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

######################################## split data into two parts: training + validation
def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

########################################
def prepare_X(raw_X):
    """Extract features from raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 256].

    Returns:
        X: An array of shape [n_samples, n_features].
    """
   
    
    raw_image = raw_X.reshape((-1, 16, 16))  ## make it 16 x 16       -1 means remain dimension
 
   
    
	# Feature 1: Measure of Symmetry
	### YOUR CODE HERE

    X = [0,0,0]

    ### split into image units
    for unit in raw_image:
        
        feature = []
        
        feature.append(1)
        
        ### split every unit
        symmetry = 0
        for i in unit:
            for j in range(15): 
                symmetry = symmetry - abs(i[j]-i[15-j])
                
        feature.append(symmetry/256)
        
        intensity = 0
        for i in unit:
            for j in i:
                intensity = intensity + j
                
        feature.append(intensity/256)     
          
        X = np.row_stack((X,feature)) 
        
   # print("------")
   #  print((X[0]))

    X = np.delete(X, (0), axis=0)
    
    return X

########################################
def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 5.
    """
    y = raw_y
    
    idx = np.where( (raw_y==1) | (raw_y==2) )
 
    print('----------------------------')
    print(int(idx[0][1]))
    
    idx_t = []
    
    i = 0
    
    print(len(idx[0]))
    
    
    for i in range(len(idx[0])): 
        idx_t.append(int(idx[0][i]))
    
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2
 
    return y, idx_t

########################################  Personal Testing Code
print("test")

ds_x, ds_y = load_data('/Users/dior/Desktop/2019_Fall_Class/636/HW/1/data/training.npz');

# print(ds_x)


# /Users/dior/Desktop/2019_Fall_Class/636/HW/1/data/training.npz 
########################################

ds_x_train, ds_x_valid, ds_y_train, ds_y_valid = train_valid_split(ds_x, ds_y, 2600)

  

print(prepare_X(ds_x_train))
# m,n = prepare_y(ds_y_train)
# print(n)
## print(prepare_y(ds_y_train))


########################################





import tensorflow as tf
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os
from numpy.linalg import inv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '8'

print('test')

#################################### PCA Testing
def test_pca(p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

#################################### AE Testing    
def test_ae(p):
    model = AE(sess=tf.Session(), d_hidden_rep=p)

    model.train(A, A, 135, 300)

#   model.train(A, A, 128 ''' batch ''', 300 ''' epoch - learning rate''')
    A_re = model.reconstruction(A)

    final_w = model.get_params()

    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

#################################### Self testing
if __name__ == '__main__':
    dataloc = "/Users/dior/Desktop/636/HW2/Prob5/data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()
    print(A.shape)
    '''
    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
    ps = [50, 100, 150]
    for p in ps:
        G = test_pca(p)
        final_w = test_ae(p)  
    ### END YOUR CODE 
    '''
    


    '''
    test_pca(32)
    test_pca(64)
    test_pca(128)
    ''' 

 
    
   
    
    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
     
      #  ps = [128]
    ps = [64]
    for p in ps:
        G = test_pca(p)
        final_w = test_ae(p)  

        a = frobeniu_norm_error(G,final_w)    
        
        print('frobeniu_norm_error is')      
        print(a)      

        ### END YOUR CODE 
     
    


"""
Parallel implementations of loss functions for NMF
"""
import numpy as np

from .parallel_operations import matrix_multiplication, matrix_subtract, matrix_square, matrix_sum

def euclidean_loss(X, W, H, compare_to_numpy=False):
    """
    sum((X - WH)^2)
    """
    
    # compute WH
    WH = matrix_multiplication(W, H)
    
    if compare_to_numpy:
        print("W x H matches NumPy: ", np.allclose(WH, np.matmul(W, H)))
    
    
    # subtract WH from X: X - WH
    X_minus_WH = matrix_subtract(X, WH)
    
    if compare_to_numpy:
        print("X - WH matches NumPy: ", np.allclose(X_minus_WH, X - np.matmul(W, H)))
    
    # square 
    X_minus_WH_squared = matrix_square(X_minus_WH)
    
    if compare_to_numpy:
        print("(X - WH)^2 matches NumPy: ", np.allclose(X_minus_WH_squared, np.square(X - np.matmul(W, H))))
    
    #sum
    result = matrix_sum(X_minus_WH_squared)
    
    if compare_to_numpy:
            print("sum((X - WH)^2) matches NumPy: ", np.allclose(result, np.sum(np.square(X - np.matmul(W, H)))))        
    
    return result
    
    
    
    
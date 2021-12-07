"""
Parallel implementations of loss functions for NMF
"""
import numpy as np
import cupy as cp
import time

from parallel_operations import matrix_multiplication, matrix_subtract, matrix_square, matrix_sum

def euclidean_loss(X, W, H, compare_to_numpy=False, return_time=False):
    """
    sum((X - WH)^2)
    """
    if return_time:
        start = time.time()
    
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
            
    if return_time:
        end = time.time()
        return result, (end-start)*1e3
    
    return result


def euclidean_loss_numpy(X, W, H, return_time=False):
    if return_time:
        start = time.time()
        
    loss = np.matrix(np.square(X - (W.dot(H)))).sum()
    
    if return_time:
        end = time.time()
        return loss, (end - start)*1e3
    
    return loss


def euclidean_loss_cupy(X, W, H, compare_to_numpy=False, return_time=False):
    if return_time:
        start = time.time()
        
    X = cp.array(X)
    W = cp.array(W)
    H = cp.array(H)
    
    loss = cp.sum(cp.square(X - cp.matmul(W, H)))
    
    if return_time:
        end = time.time()
        return loss, (end - start)*1e3
    
    return loss


def divergence_loss_numpy(X, W, H, eps, return_time=False):
    if return_time:
        start = time.time()
        
    loss = np.sum(-X * np.log(W.dot(H)+eps) + W.dot(H))
    
    if return_time:
        end = time.time()
        return loss, (end - start)*1e3
    
    return loss
    
    



    
    
    
    
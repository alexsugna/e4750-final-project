"""
Parallel implementations of loss functions for NMF
"""
import numpy as np
import cupy as cp
import time
from pycuda import gpuarray
import pycuda.driver as cuda


from .parallel_operations import matrix_multiplication, matrix_subtract, matrix_square, matrix_sum

def euclidean_loss_parallel(X_d, WH_d, N, M, context, src_mod, compare_to_numpy=False, return_time=False):
    """
    sum((X - WH)^2) TODO: docstring
    """
    X_minus_WH_d = gpuarray.zeros(((N, M)), dtype=np.float32)
    X_minus_WH_d_square = gpuarray.zeros(((N, M)), dtype=np.float32)
    
    func_sub = src_mod.get_function("MatEleSubtract") # matrix elementwise subtraction
    func_sqr = src_mod.get_function("MatEleSquare")

    block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, M)
    event = func_sub(X_d, WH_d, X_minus_WH_d, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()

    event = func_sqr(X_minus_WH_d, X_minus_WH_d_square, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
    cuda.Context.synchronize()

    result = matrix_sum(X_minus_WH_d_square.get())

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
    
    



    
    
    
    
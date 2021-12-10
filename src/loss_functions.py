"""
Parallel and serial implementations of loss functions for NMF.
"""

import time
import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda

from .parallel_operations import matrix_sum

def euclidean_loss_parallel(X_d, WH_d, N, M, context, src_mod, return_time=False):
    """
    Parallel Euclidean Loss function calculation.

    sum((X - WH)^2)

    params:
        X_d (gpuarray): X matrix on device. Size (N, M)

        WH_d (gpuarray): result of matrix multiplication between W and H on device. Size (N, M)

        N (int): number of rows in X

        M (int): number of columns in X

        context (Context): instance of class "Context" defined in src/context.py

        src_mod (pycuda.compiler.SourceModule): Instance of pycuda.compiler.SourceModule
                                                with CUDA kernels loaded.

        return_time=False (bool): if True, function returns execution time in seconds.

    returns:
        loss (float): euclidean loss of X and WH calculated in parallel.
    """
    if return_time:
        start = time.time() # record start time

    # initialize intermediate computation buffer arrays
    X_minus_WH_d = gpuarray.zeros(((N, M)), dtype=np.float32)
    X_minus_WH_d_square = gpuarray.zeros(((N, M)), dtype=np.float32)

    # define kernel functions
    func_sub = src_mod.get_function("MatEleSubtract") # matrix elementwise subtraction
    func_sqr = src_mod.get_function("MatEleSquare") # matrix elementwise square

    # calculate block and grid dimensions
    block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, M)
    func_sub(X_d, WH_d, X_minus_WH_d, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim) # call CUDA subtraction kernel
    cuda.Context.synchronize() # wait for subtraction to finish

    func_sqr(X_minus_WH_d, X_minus_WH_d_square, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim) # call CUDA square kernel
    cuda.Context.synchronize() # wait for square to finish

    loss = matrix_sum(X_minus_WH_d_square.get()) # call parallel maatri sum funciton

    if return_time: # record end time
        end = time.time()
        return loss, (end - start)*1e3 # return loss and execution time in s

    # return the Euclidean loss
    return loss


def divergence_loss_parallel():
    print("NOT IMPLEMENTED!")


def euclidean_loss_numpy(X, W, H, return_time=False):
    """
    Serial (NumPy) Euclidean loss calculation.

    sum((X - WH)^2)

    params:
        X (np.array): The data matrix X of shape (N, M)

        W (np.array): The W matrix of shape (N, K)

        H (np.array): The H matrix of shape (K, M)

        return_time=False (bool): if True, function returns execution time in seconds.

    returns:
        loss (float): euclidean loss of X and WH calculated serially.
    """
    if return_time:
        start = time.time() # record start time

    loss = np.matrix(np.square(X - (W.dot(H)))).sum() # calculate loss sum((X - WH)^2)

    if return_time: # record end time
        end = time.time()
        return loss, (end - start)*1e3 # return loss and execution time in s

    return loss


def divergence_loss_numpy(X, W, H, eps, return_time=False):
    """
    Serial (NumPy) Divergence loss calculation.

    sum(-X * log(WH) + WH)

    params:
        X (np.array): The data matrix X of shape (N, M)

        W (np.array): The W matrix of shape (N, K)

        H (np.array): The H matrix of shape (K, M)

        eps (float): small value added in loss calculation for numerical stability

        return_time=False (bool): if True, function returns execution time in seconds.

    returns:
        loss (float): divergence loss of X and WH calculated serially.
    """
    if return_time:
        start = time.time() # record start time

    loss = np.sum(-X * np.log(W.dot(H)+eps) + W.dot(H)) # calculate loss sum(-X * log(WH) + WH)

    if return_time: # record end time
        end = time.time()
        return loss, (end - start)*1e3 # return loss and execution time in s

    return loss

"""
Our serial implementation of NMF
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from pycuda import gpuarray

from .loss_functions import euclidean_loss_numpy, divergence_loss_numpy, euclidean_loss_parallel
from .context import Context

BLOCK_SIZE = 32

def NMF_serial(X, W, H, iterations=100, loss='euclidean', eps=1e-16, return_time=True, print_iterations=True):
    """
    Performs NumPy (serial) NMF.
    
    params:
        X (N, M): the original data matrix
        
        W (N, K): the W matrix (of articles by topic)
        
        H (K, M): the H matrix (of topics by word)
        
        iterations=100 (int): the number of matrix factorization updates
        
        loss='euclidean' (string): one of ['euclidean', 'divergence'] to specify the loss function
        
        eps=1e-16 (float): small value epsilon added for numerical stability
        
        return_time=True (bool): return the execution time in s
        
    returns:
        W (N, K): the factored matrix W
        
        H (K, M): the factored matrix H
        
        losses (list): List of loss at each iteration
    """
    if return_time:
        start = time.time()
        
    losses = [] #keep track of objective function evaluation for each iteration
    
    for i in range(1, iterations+1):
        if i % 10 == 0 and print_iterations:
            print('iteration %d' % i)

        #objective function
        if loss == 'euclidean':
            
            Wt = W.T #w transpose
            H = H * Wt.dot(X) / (Wt.dot(W).dot(H) + eps) #update H

            Ht = H.T #H transpose
            W = W * X.dot(Ht) / (W.dot(H).dot(Ht) + eps) #update W
            
            iter_loss = euclidean_loss_numpy(X, W, H)
            
        elif loss == 'divergence': 
            
            P = X / (W.dot(H)+eps) #intermediate step (purple matrix in notes)
            Wt = W.T
            Wt = Wt / Wt.sum(axis=1).reshape(-1, 1) #normalize rows
            H = H * (Wt.dot(P))  #update H
            P = X / (W.dot(H)+eps)
            Ht = H.T
            Ht = Ht / Ht.sum(axis=0).reshape(1, -1) #normalize columns
            W = W * (P.dot(Ht))  #update W
            
            iter_loss = divergence_loss_numpy(X, W, H, eps)

        else:
            raise Exception('Loss function "{}" not supported.'.format(loss))
            
        losses.append(iter_loss)
        
    if return_time:
        end = time.time()
        return W, H, losses, end - start 
    
    return W, H, losses


def NMF_parallel(X, W, H, iterations=100, loss='euclidean', eps=1e-16, return_time=True, print_iterations=True):
    """
    Performs CUDA (parallel) NMF.
    
    params:
        X (N, M): the original data matrix
        
        W (N, K): the W matrix (of articles by topic)
        
        H (K, M): the H matrix (of topics by word)
        
        iterations=100 (int): the number of matrix factorization updates
        
        loss='euclidean' (string): one of ['euclidean', 'divergence'] to specify the loss function
        
        eps=1e-16 (float): small value epsilon added for numerical stability
        
        return_time=True (bool): return the execution time in s
        
    returns:
        W (N, K): the factored matrix W
        
        H (K, M): the factored matrix H
        
        squared_out (list): List of loss at each iteration
    """
    context = Context(BLOCK_SIZE) # define context
    
    # define kernel paths
    kernel_paths = ['./kernels/MatrixMultiplication.cu',
                    './kernels/MatrixTranspose.cu',
                    './kernels/ElementWise.cu']
    
    src_mod = context.getSourceModule(kernel_paths, multiple_kernels=True)
    
    func_mul = src_mod.get_function("MatMul") # matrix multiplication
    func_ele_mul = src_mod.get_function("MatEleMulInPlace") # matrix multiplication elementwise
    func_tran = src_mod.get_function("MatTran") # matrix transpose
    func_add = src_mod.get_function("MatEleAddInPlace") # matrix elementwise addition
    func_div = src_mod.get_function("MatEleDivideInPlace") # matrix elementwise division
    
    if return_time:
        # Event objects to mark start and end points
        g_start = cuda.Event()
        g_end = cuda.Event()
        g_start.record()
        
    # define matrix shapes
    K = W.shape[1]
    N, M = X.shape
        
    #define X, W, and H on gpu
    X_d = gpuarray.to_gpu(X)
    W_d = gpuarray.to_gpu(W)
    H_d = gpuarray.to_gpu(H)
    
    #define intermediate steps on gpu for H update
    Wt_d = gpuarray.zeros(((K, N)), dtype=np.float32)
    WtX_d = gpuarray.zeros(((K, M)), dtype=np.float32)
    WtW_d = gpuarray.zeros(((K, K)), dtype=np.float32)
    WtWH_d = gpuarray.zeros(((K, M)), dtype=np.float32)
    
    #itermediate steps for W update
    Ht_d = gpuarray.zeros(((M, K)), dtype=np.float32)
    WH_d = gpuarray.zeros(((N, M)), dtype=np.float32)
    WHHt_d = gpuarray.zeros(((N, K)), dtype=np.float32)
    XHt_d = gpuarray.zeros(((N, K)), dtype=np.float32)
    
    # Record execution time and execute operation.
    for i in range(1, iterations+1):
        if i % 10 == 0 and print_iterations:
            print('iteration %d' % i)
            
        #UPDATE H *****************************************************************************
        #Wt = W.T
        #H = H * Wt.dot(X) / (Wt.dot(W).dot(H) + err)
        
        #Wt = W.T
        block_dim, grid_dim = context.block_dims, context.grid_dims2d(K,N)
        event = func_tran(W_d, Wt_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()
        
        #Wt * X = WtX
        block_dim, grid_dim = context.block_dims, context.grid_dims2d(K,M)
        event = func_mul(Wt_d, X_d, WtX_d, np.int32(K), np.int32(N), np.int32(N), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
    
        #Wt * W = WtW
        block_dim, grid_dim = context.block_dims, context.grid_dims2d(K,K)
        event = func_mul(Wt_d, W_d, WtW_d, np.int32(K), np.int32(N), np.int32(N), np.int32(K), np.int32(K), np.int32(K), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #WtW * H = WtWH
        block_dim, grid_dim = context.block_dims, context.grid_dims2d(K,M)
        event = func_mul(WtW_d, H_d, WtWH_d, np.int32(K), np.int32(K), np.int32(K), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #WtWH + err
        event = func_add(WtWH_d, np.float32(eps), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #H .* WtX elementwise
        event = func_ele_mul(H_d, WtX_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #H / WtWH elementwise
        event = func_div(H_d, WtWH_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()
        
        ##### W UPDATE #####

        #Ht = H.T #H transpose
        event = func_tran(H_d, Ht_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #X * Ht = XHt
        block_dim, grid_dim = context.block_dims, context.grid_dims2d(N,K)
        event = func_mul(X_d, Ht_d, XHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()            

        #W * H = WH
        block_dim, grid_dim = context.block_dims, context.grid_dims2d(N,M)
        event = func_mul(W_d, H_d, WH_d, np.int32(N), np.int32(K), np.int32(K), np.int32(M), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #WH * Ht = WHHt
        block_dim, grid_dim = context.block_dims, context.grid_dims2d(N,K)
        event = func_mul(WH_d, Ht_d, WHHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #WHHt + err
        event = func_add(WHHt_d, np.float32(eps), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #W .* XHt elementwise
        event = func_ele_mul(W_d, XHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #W / WHHt elementwise
        event = func_div(W_d, WHHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        #W = W * X.dot(Ht) / (W.dot(H).dot(Ht) + err) #update W
        
#         euclidean_loss = euclidean_loss_parallel(X_d, WH_d, N, M, context, src_mod)
#         euclidean_loss_np = euclidean_loss_numpy(X_d.get(), W_d.get(), H_d.get())
        
    #g_end.record()
    # Fetch result from device to host
    H = H_d.get()
    W = W_d.get()
    
    if return_time:
        g_end.record()
        g_end.synchronize()
        
        total_time = g_start.time_till(g_end)*1e-3
        
        return W, H, total_time
    
    return W, H


def plot_loss(loss, title, xlab="Iterations", ylab="Loss", loss_type='Euclidean Distance'):
    """
    Makes a plot of the given loss history.
    """
    fig = plt.figure()
    plt.plot(np.arange(1, len(loss) + 1), loss, label=loss_type)
    plt.grid()
    plt.legend()
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title)
    plt.savefig('{}.png'.format(title))
        
    
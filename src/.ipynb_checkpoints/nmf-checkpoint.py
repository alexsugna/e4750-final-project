"""
Our serial implementation of NMF
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from pycuda import gpuarray
from sklearn.decomposition import NMF

from .loss_functions import euclidean_loss_numpy, divergence_loss_numpy, euclidean_loss_parallel
from .context import Context

BLOCK_SIZE = 32

def NMF_serial(X, W, H, iterations=100, loss='euclidean', eps=1e-16, return_time=True, print_iterations=True, calculate_loss=True, print_start=True):
    """
    Performs NumPy (serial) NMF.

    params:
        X (N, M): the original data matrix

        W (N, K): the W matrix (of articles by topic)

        H (K, M): the H matrix (of topics by word)

        iterations=100 (int): the number of matrix factorization updates

        loss='euclidean' (string): one of ['euclidean', 'divergence'] to specify the loss function/update scheme

        eps=1e-16 (float): small value epsilon added for numerical stability

        return_time=True (bool): return the execution time in s
        
        calculate_loss=True (bool): calculates loss at each iteration and returns list. 
        
        print_start=True (bool): print string indicating start of NMF

    returns:
        W (N, K): the factored matrix W

        H (K, M): the factored matrix H

        losses (list): List of loss at each iteration
    """
    if print_start:
        print("Starting {} iterations of serial NMF with {} loss.".format(iterations, loss)) 
    
    if return_time:
        start = time.time()

    losses = [] #keep track of objective function evaluation for each iteration

    for i in range(1, iterations+1):
        if i % 10 == 0 and print_iterations:
            print('Iteration: %d' % i)

        #objective function
        if loss == 'euclidean':

            Wt = W.T #w transpose
            H = H * Wt.dot(X) / (Wt.dot(W).dot(H) + eps) #update H

            Ht = H.T #H transpose
            W = W * X.dot(Ht) / (W.dot(H).dot(Ht) + eps) #update W
            
            if calculate_loss:
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
            
            if calculate_loss:
                iter_loss = divergence_loss_numpy(X, W, H, eps)

        else:
            raise Exception('Loss function "{}" not supported.'.format(loss))
            
        if calculate_loss:
            losses.append(iter_loss)

    if return_time:
        end = time.time()
        return W, H, losses, end - start

    return W, H, losses


def NMF_parallel(X, W, H, iterations=100, loss='euclidean', eps=1e-16, return_time=True, print_iterations=True,
                 print_start=True):
    """
    Performs CUDA (parallel) NMF.

    params:
        X (N, M): the original data matrix

        W (N, K): the W matrix (of articles by topic)

        H (K, M): the H matrix (of topics by word)

        iterations=100 (int): the number of matrix factorization updates

        loss='euclidean' (string): one of ['euclidean', 'divergence'] to specify the loss function/update scheme

        eps=1e-16 (float): small value epsilon added for numerical stability

        return_time=True (bool): return the execution time in s
        
        print_start=True (bool): print string indicating start of NMF

    returns:
        W (N, K): the factored matrix W

        H (K, M): the factored matrix H

        squared_out (list): List of loss at each iteration
    """
    if print_start:
        print("Starting {} iterations of parallel NMF with {} multiplicative updates.".format(iterations, loss)) 
        
    context = Context(BLOCK_SIZE) # define context

    # define kernel paths
    kernel_paths = ['./kernels/MatrixMultiplication.cu',
                    './kernels/MatrixTranspose.cu',
                    './kernels/ElementWise.cu',
                    './kernels/RowColSum.cu']

    src_mod = context.getSourceModule(kernel_paths, multiple_kernels=True)

    func_mul = src_mod.get_function("MatMul") # matrix multiplication
    func_ele_mul = src_mod.get_function("MatEleMulInPlace") # matrix multiplication elementwise
    func_tran = src_mod.get_function("MatTran") # matrix transpose
    func_add = src_mod.get_function("MatEleAddInPlace") # matrix elementwise addition
    func_div = src_mod.get_function("MatEleDivideInPlace") # matrix elementwise division
    func_divC = src_mod.get_function("MatEleDivide") # gives output C matrix
    func_row_sum = src_mod.get_function("row_sum") # matrix elementwise division
    func_col_sum = src_mod.get_function("column_sum") # matrix elementwise division
    func_row_div = src_mod.get_function("MatEleDivideRow") # divide row by sum
    func_col_div = src_mod.get_function("MatEleDivideCol") # divide col by sum

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
    
    if loss == 'euclidean':
        """
        Begin Euclidean Update
        """
        #define intermediate buffers on gpu for H update
        Wt_d = gpuarray.zeros(((K, N)), dtype=np.float32)
        WtX_d = gpuarray.zeros(((K, M)), dtype=np.float32)
        WtW_d = gpuarray.zeros(((K, K)), dtype=np.float32)
        WtWH_d = gpuarray.zeros(((K, M)), dtype=np.float32)

        # define itermediate buffers for W update
        Ht_d = gpuarray.zeros(((M, K)), dtype=np.float32)
        WH_d = gpuarray.zeros(((N, M)), dtype=np.float32)
        WHHt_d = gpuarray.zeros(((N, K)), dtype=np.float32)
        XHt_d = gpuarray.zeros(((N, K)), dtype=np.float32)
        
        # begin for loop for multiplicative updates
        for i in range(1, iterations+1):
            if i % 10 == 0 and print_iterations:
                print('Iteration: %d' % i)

            """
            Begin Euclidean Update H
            """
            # Wt = W.T
            # H = H * Wt.dot(X) / (Wt.dot(W).dot(H) + eps)

            # Wt = W.T
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, K)
            event = func_tran(W_d, Wt_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # Wt * X = WtX
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(K,M)
            event = func_mul(Wt_d, X_d, WtX_d, np.int32(K), np.int32(N), np.int32(N), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)

            # Wt * W = WtW
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(K,K)
            event = func_mul(Wt_d, W_d, WtW_d, np.int32(K), np.int32(N), np.int32(N), np.int32(K), np.int32(K), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # WtW * H = WtWH
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(K,M)
            event = func_mul(WtW_d, H_d, WtWH_d, np.int32(K), np.int32(K), np.int32(K), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # WtWH + eps
            event = func_add(WtWH_d, np.float32(eps), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # H .* WtX elementwise
            event = func_ele_mul(H_d, WtX_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # H / WtWH elementwise
            event = func_div(H_d, WtWH_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            """
            End Euclidean Update H
            """
            """
            Begin Euclidean Update W
            """
            # Ht = H.T #H transpose
            event = func_tran(H_d, Ht_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # X * Ht = XHt
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N,K)
            event = func_mul(X_d, Ht_d, XHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # W * H = WH
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N,M)
            event = func_mul(W_d, H_d, WH_d, np.int32(N), np.int32(K), np.int32(K), np.int32(M), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # WH * Ht = WHHt
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N,K)
            event = func_mul(WH_d, Ht_d, WHHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # WHHt + eps
            event = func_add(WHHt_d, np.float32(eps), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # W .* XHt elementwise
            event = func_ele_mul(W_d, XHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # W / WHHt elementwise
            event = func_div(W_d, WHHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            """
            End Euclidean Update W
            """
        """
        End Euclidean Update
        """
    
    elif loss == 'divergence':
        """
        Begin Divergence Update
        """
        
        # define intermediate buffers for H update
        WH_d = gpuarray.zeros(((N,M)), dtype=np.float32)
        P_d = gpuarray.zeros(((N,M)), dtype=np.float32)
        Wt_d = gpuarray.zeros(((K,N)), dtype=np.float32)
        Wt_sum_d = gpuarray.zeros(((K,1)), dtype=np.float32) #sum rows
        WtP_d = gpuarray.zeros(((K,M)), dtype=np.float32)

        # define intermediate buffers on gpu for W update
        Ht_d = gpuarray.zeros(((M,K)), dtype=np.float32)
        Ht_sum_d = gpuarray.zeros(((1,K)), dtype=np.float32) #sum cols
        PHt_d = gpuarray.zeros(((N,K)), dtype=np.float32)
        
        # begin for loop for multiplicative updates
        for i in range(1, iterations+1):
            if i % 10 == 0 and print_iterations:
                print('Iteration: %d' % i)
            
            """
            Begin Divergence Update H
            """
            # P = X / (W.dot(H)+eps) #intermediate step (purple matrix in notes)
            # Wt = W.T
            # Wt = Wt / Wt.sum(axis=1).reshape(-1, 1) #normalize rows
            # H = H * (Wt.dot(P))  #update H
            
            # W.dot(H)
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, M)
            event = func_mul(W_d, H_d, WH_d, np.int32(N), np.int32(K), np.int32(K), np.int32(M), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # WH + eps
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, M)
            event = func_add(WH_d, np.float32(eps), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # X / WH (saved as P_d)
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, M)
            event = func_divC(X_d, WH_d, P_d, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # Wt = W.T
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, K)
            event = func_tran(W_d, Wt_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # Wt.sum(axis=1).reshape(-1, 1) #sum rows
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(K, 1)
            event = func_row_sum(Wt_d, Wt_sum_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize() 

            # Wt = Wt / Wt_sum_d #elementwise
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(K, N)
            event = func_row_div(Wt_d, Wt_sum_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # Wt.dot(P)
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(K, M)
            event = func_mul(Wt_d, P_d, WtP_d, np.int32(K), np.int32(N), np.int32(N), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            
            # H .* WtP elementwise
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(K, M)
            event = func_ele_mul(H_d, WtP_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            """
            End Divergence Update H
            """
            """
            Begin Divergence Update W
            """
            # P = X / (W.dot(H)+eps)
            # Ht = H.T
            # Ht = Ht / Ht.sum(axis=0).reshape(1, -1) #normalize columns
            # W = W * (P.dot(Ht))  #update W
            
            # W.dot(H)
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, M)
            event = func_mul(W_d, H_d, WH_d, np.int32(N), np.int32(K), np.int32(K), np.int32(M), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            
            # WH + eps
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, M)
            event = func_add(WH_d, np.float32(eps), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # X / WH (saved as P_d)
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, M)
            event = func_divC(X_d, WH_d, P_d, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # Ht = H.T #H transpose
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(K, M)
            event = func_tran(H_d, Ht_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # Ht.sum(axis=0).reshape(1, -1) #sum columns
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(1, K)
            event = func_col_sum(Ht_d, Ht_sum_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # Ht = Ht / Ht_sum_d #elementwise
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(M, K)
            event = func_col_div(Ht_d, Ht_sum_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # P.dot(Ht)
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, K)
            event = func_mul(P_d, Ht_d, PHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            # W .* PHt elementwise
            block_dim, grid_dim = context.block_dims, context.grid_dims2d(N, K)
            event = func_ele_mul(W_d, PHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            """
            End Divergence Update W
            """
        """
        End Divergence Update
        """
        
    else:
        raise Exception('Loss function "{}" not supported.'.format(loss))
        
           
    # Fetch result from device to host
    H = H_d.get()
    W = W_d.get()

    if return_time:
        g_end.record()
        g_end.synchronize()

        total_time = g_start.time_till(g_end)*1e-3 # record total time in seconds

        return W, H, total_time

    return W, H


def NMF_sklearn(X, W, H, iterations=100, loss='euclidean', return_time=True, print_start=True):
    """
    A wrapper function for sklearn.decomposition.NMF. The purpose of this function is to 
    match to function call signature of NMF_serial() and NMF_parallel() above to simplify
    the execution time comparison code.
    
    params:
        X (N, M): the original data matrix

        W (N, K): the W matrix (of articles by topic)

        H (K, M): the H matrix (of topics by word)

        iterations=100 (int): the number of matrix factorization updates

        loss='euclidean' (string): one of ['euclidean', 'divergence'] to specify the loss function/update scheme

        return_time=True (bool): return the execution time in s
        
        print_start=True (bool): print string indicating start of NMF

    returns:
        W (N, K): the factored matrix W

        H (K, M): the factored matrix H
    """
    if print_start:
        print("Starting {} iterations of Scikit-learn NMF with {} loss.".format(iterations, loss)) 
    
    K = W.shape[1] # get num_topics
    
    if loss == 'euclidean': # map loss type to string sklearn is expecting
        beta_loss = 'frobenius'
        
    elif loss == 'divergence':
        beta_loss = 'kullback-leibler'
        
    else:
        raise Exception('Loss function "{}" not supported.'.format(loss))
        
    if return_time:
        start = time.time() # record start time
        
    model = NMF(n_components=K, init='custom', beta_loss=beta_loss, max_iter=iterations, solver='mu', tol=1e-10)
    
    W_sklearn = model.fit_transform(X, W=W, H=H) # retrieve W, H
    H_sklearn = model.components_
    
    if return_time: # return execution time and W, H
        end = time.time()
        return W_sklearn, H_sklearn, end - start
    
    return W_sklearn, H_sklearn
    
    

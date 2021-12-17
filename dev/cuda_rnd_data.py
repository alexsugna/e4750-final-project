import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from pycuda import gpuarray
import time as timer

import pandas as pd
import numpy as np
from matplotlib import pyplot
import math
from scipy.sparse import random

class cudaNMF:
    def __init__(self):
        
        # define block and grid dimensions
        
        # Compile the kernel code when an instance of this class is made.
        self.mod = self.getSourceModule()

    def getGridDimention(self, x, y):
        """
        Get the appropriate grid dimention based on 
        """
        blockDim = (32,32,1) #1x1024 blocks
        gridDim = (math.ceil(y / 32),math.ceil(x / 32), 1)
        return blockDim, gridDim

    def getSourceModule(self):  
        #open first function
        text_file = open("/home/sls2305/Desktop/e4750-final-project-main/kernels/MatrixMultiplication.cu", "r")
        func1 = text_file.read()
        text_file.close()
        
        #open second function
        text_file = open("/home/sls2305/Desktop/e4750-final-project-main/kernels/MatrixTranspose.cu", "r")
        func2 = text_file.read()
        text_file.close()

        text_file = open("/home/sls2305/Desktop/e4750-final-project-main/kernels/ElementWise.cu", "r")
        func3 = text_file.read()
        text_file.close()

        kernelwrapper = str(func1) + str(func2) + str(func3);
        return SourceModule(kernelwrapper)

    def MatMul(self, a,b):

        ACols = a.shape[1] #get number of columns of input A
        ARows = a.shape[0] #get number of rows of input A
        BCols = b.shape[1] #get number of columns of input B
        BRows = b.shape[0] #get number of rows of input B
        #output dimensions
        CCols = BCols
        CRows = ARows

        # Get kernel function
        func = self.mod.get_function("MatMul")

        # Device memory allocation for input and output array(s)
        c = np.zeros( ((CRows,CCols)), dtype=np.float32)
        
        a_d = gpuarray.to_gpu(a)
        b_d = gpuarray.to_gpu(b)
        c_d = gpuarray.to_gpu(c)

        # Record execution time and execute operation.
        block_dim, grid_dim = self.getGridDimention(CRows,CCols)
        print(block_dim)
        print(grid_dim)

        event = func(a_d, b_d, c_d, np.int32(ARows), np.int32(ACols), np.int32(BRows),np.int32(BCols), np.int32(CRows), np.int32(CCols), block=block_dim, grid=grid_dim)
        
        # Wait for the event to complete
        cuda.Context.synchronize()

        # Fetch result from device to host
        c = c_d.get()

        # Convert output array back to string

        return c #, time_

    def MatTran(self, a):

        ACols = a.shape[1] #get number of columns of input A
        ARows = a.shape[0] #get number of rows of input A
        
        # Get kernel function
        func = self.mod.get_function("MatTran")

        # Device memory allocation for input and output array(s)
        c = np.zeros( ((ACols,ARows)), dtype=np.float32)
        
        a_d = gpuarray.to_gpu(a)
        c_d = gpuarray.to_gpu(c)

        # Record execution time and execute operation.
        block_dim, grid_dim = self.getGridDimention(ARows,ACols)
        

        event = func(a_d, c_d, np.int32(ACols), np.int32(ARows), block=block_dim, grid=grid_dim)
        
        # Wait for the event to complete
        cuda.Context.synchronize()

        # Fetch result from device to host
        c = c_d.get()

        # Convert output array back to string

        return c #, time_
    
    def NMF(self, X, N, M, K, W, H):

        #ACols = a.shape[1] #get number of columns of input A
        #ARows = a.shape[0] #get number of rows of input A
        
        # Get kernel function
        func_mul = self.mod.get_function("MatMul") #matrix multiplication
        func_ele_mul = self.mod.get_function("MatEleMulInPlace") #matrix multiplication elementwise
        func_tran = self.mod.get_function("MatTran") #matrix transpose
        func_add = self.mod.get_function("MatEleAddInPlace") #matrix elementwise addition
        func_div = self.mod.get_function("MatEleDivideInPlace") #matrix elementwise division
        
        # Event objects to mark start and end points
        g_start = cuda.Event()
        g_end = cuda.Event()
        g_start.record()

        #define X, W, and H on gpu
        X_d = gpuarray.to_gpu(X)
        W_d = gpuarray.to_gpu(W)
        H_d = gpuarray.to_gpu(H)
    
        #define intermediate steps on gpu for H update
        Wt_d = gpuarray.zeros(((K,N)), dtype=np.float32)
        WtX_d = gpuarray.zeros(((K,M)), dtype=np.float32)
        WtW_d = gpuarray.zeros(((K,K)), dtype=np.float32)
        WtWH_d = gpuarray.zeros(((K,M)), dtype=np.float32)

        #itermediate steps for W update
        Ht_d = gpuarray.zeros(((M,K)), dtype=np.float32)
        WH_d = gpuarray.zeros(((N,M)), dtype=np.float32)
        WHHt_d = gpuarray.zeros(((N,K)), dtype=np.float32)
        XHt_d = gpuarray.zeros(((N,K)), dtype=np.float32)

        #get block dim and grid dim
        
        err = 1e-16 #a small error to prevent 0/0
        iteration = 100 #number of iterations
        
        

        # Record execution time and execute operation.
        for i in range(1, iteration+1):
            if i % 10 == 0:
                print('iteration %d' % i)
            
            #UPDATE H *****************************************************************************
            #Wt = W.T
            #H = H * Wt.dot(X) / (Wt.dot(W).dot(H) + err)

            #Wt = W.T
            block_dim, grid_dim = self.getGridDimention(K,N)
            event = func_tran(W_d, Wt_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #Wt * X = WtX
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_mul(Wt_d, X_d, WtX_d, np.int32(K), np.int32(N), np.int32(N), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize() #<--maybe remove

            #Wt * W = WtW
            block_dim, grid_dim = self.getGridDimention(K,K)
            event = func_mul(Wt_d, W_d, WtW_d, np.int32(K), np.int32(N), np.int32(N), np.int32(K), np.int32(K), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WtW * H = WtWH
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_mul(WtW_d, H_d, WtWH_d, np.int32(K), np.int32(K), np.int32(K), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WtWH + err
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_add(WtWH_d, np.float32(err), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #H .* WtX elementwise
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_ele_mul(H_d, WtX_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #H / WtWH elementwise
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_div(H_d, WtWH_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W UPDATE*************************************************************************************

            #Ht = H.T #H transpose
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_tran(H_d, Ht_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #X * Ht = XHt
            block_dim, grid_dim = self.getGridDimention(N,K)
            event = func_mul(X_d, Ht_d, XHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()            

            #W * H = WH
            block_dim, grid_dim = self.getGridDimention(N,M)
            event = func_mul(W_d, H_d, WH_d, np.int32(N), np.int32(K), np.int32(K), np.int32(M), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WH * Ht = WHHt
            block_dim, grid_dim = self.getGridDimention(N,K)
            event = func_mul(WH_d, Ht_d, WHHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WHHt + err
            block_dim, grid_dim = self.getGridDimention(N,K)
            event = func_add(WHHt_d, np.float32(err), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W .* XHt elementwise
            block_dim, grid_dim = self.getGridDimention(N,K)
            event = func_ele_mul(W_d, XHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W / WHHt elementwise
            block_dim, grid_dim = self.getGridDimention(N,K)
            event = func_div(W_d, WHHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W = W * X.dot(Ht) / (W.dot(H).dot(Ht) + err) #update W
        
        #g_end.record()
        # Fetch result from device to host
        H = H_d.get()
        W = W_d.get()

        g_end.record()
        g_end.synchronize()

        return H,W,g_start.time_till(g_end)*1e-3


if __name__ == "__main__":
    # Main code
    test = cudaNMF()
    
    num_runs = 4
    parallel_times = np.empty(num_runs)
    serial_times = np.empty(num_runs) 
    input_size = np.array([10,100,1000,10000]) 

    for j in range(num_runs):

        #NMF test data set
        N = input_size[j] #number of words
        M = input_size[j] #number of articles
        K = 25 #rank to factor into

        #X = np.float32(np.random.rand(N, M)) #define the original data matrix
        sparse_X = random(N, M, density=0.05,dtype=np.float32) #define original sparse data
        X = sparse_X.A.astype(np.float32)

        # Device memory allocation for input and output array(s)
        W = np.float32(np.random.uniform(1, 2, (N, K))) #initialize W and H to random values between 1 and 2
        H = np.float32(np.random.uniform(1, 2, (K, M)))

        #parallel implementation***************************************************************************
        #call NMF function
        H_out,W_out,time = test.NMF(np.float32(X), N, M, K, np.float32(W), np.float32(H))
        parallel_times[j] = time

        W_out = W_out / W_out.sum(axis=0).reshape(1,-1) #normalize the 25 categories

        #numpy implementation******************************************************************************
        iteration = 100
        err = 1e-16 #a small error to prevent 0/0

        start = timer.time() #record start time

        for i in range(1, iteration+1):
            if i % 10 == 0:
                print('iteration %d' % i)
                
            Wt = W.T #w transpose
            H = (H * (Wt.dot(X))) / (((Wt.dot(W)).dot(H)) + err) #update H

            Ht = H.T #H transpose
            W = (W * (X.dot(Ht))) / (((W.dot(H)).dot(Ht)) + err) #update W

        end = timer.time() #record end time
        serial_times[j] = end-start        

        W = W / W.sum(axis=0).reshape(1,-1) #normalize the 25 categories

    pyplot.plot(input_size*input_size, serial_times, label = "serial")
    pyplot.plot(input_size*input_size, parallel_times, label = "parallel")
    
    #format
    pyplot.yscale('log')
    pyplot.xscale('log')
    pyplot.legend(loc=2, prop={'size': 9})
    pyplot.xlabel('Matrix Elements [log scale]')
    pyplot.ylabel('Runtime [log scale (sec)]')
    pyplot.title('pyCUDA Non-Negative Matrix Factorization (K=25)')
    pyplot.savefig('cuda.png')

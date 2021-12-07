import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from pycuda import gpuarray
import time as timer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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

        ACols = a.shape[1] #get number of columns of input A
        ARows = a.shape[0] #get number of rows of input A
        
        # Get kernel function
        func_mul = self.mod.get_function("MatMul") #matrix multiplication
        func_ele_mul = self.mod.get_function("MatEleMul2") #matrix multiplication elementwise
        func_tran = self.mod.get_function("MatTran") #matrix transpose
        func_add = self.mod.get_function("MatEleAdd2") #matrix elementwise addition
        func_div = self.mod.get_function("MatEleDivide2") #matrix elementwise division
        
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
        block_dim, grid_dim = self.getGridDimention(N,M)
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
            event = func_tran(W_d, Wt_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #Wt * X = WtX
            event = func_mul(Wt_d, X_d, WtX_d, np.int32(K), np.int32(N), np.int32(N), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize() #<--maybe remove

            #Wt * W = WtW
            event = func_mul(Wt_d, W_d, WtW_d, np.int32(K), np.int32(N), np.int32(N), np.int32(K), np.int32(K), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WtW * H = WtWH
            event = func_mul(WtW_d, H_d, WtWH_d, np.int32(K), np.int32(K), np.int32(K), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WtWH + err
            event = func_add(WtWH_d, np.float32(err), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #H .* WtX elementwise
            event = func_ele_mul(H_d, WtX_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #H / WtWH elementwise
            event = func_div(H_d, WtWH_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W UPDATE*******************************************************************************************

            #Ht = H.T #H transpose
            event = func_tran(H_d, Ht_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #X * Ht = XHt
            event = func_mul(X_d, Ht_d, XHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()            

            #W * H = WH
            event = func_mul(W_d, H_d, WH_d, np.int32(N), np.int32(K), np.int32(K), np.int32(M), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WH * Ht = WHHt
            event = func_mul(WH_d, Ht_d, WHHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WHHt + err
            event = func_add(WHHt_d, np.float32(err), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W .* XHt elementwise
            event = func_ele_mul(W_d, XHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W / WHHt elementwise
            event = func_div(W_d, WHHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W = W * X.dot(Ht) / (W.dot(H).dot(Ht) + err) #update W
        

        # Fetch result from device to host
        H = H_d.get()
        W = W_d.get()

        g_end.record()
        g_end.synchronize()

        return H,W,g_start.time_till(g_end)*1e-3


if __name__ == "__main__":
    # Main code
    test = cudaNMF()
    a = np.random.randint(5, size=(32, 64))
    b = np.random.randint(5, size=(64, 32))
    print("a:",a)
    print("b:",b)
    #multiplication test case
    result = test.MatMul(np.float32(a),np.float32(b))
    #print("multiply result:",result)
    serial_mul = np.matmul(a,b) #numpy multiply test case
    #print("serial multiply result:",serial_mul)
    if((result == serial_mul).all()):
        print("multiply test case passed")

    #transpose test case
    result = test.MatTran(np.float32(a))
    #print("transpose result:",result)
    serial_tran = np.transpose(a)
    if((result == serial_tran).all()):
        print("transpose test case passed")
    
    #NMF test data set
    data_path = 'nyt_data.txt'
    vocab_path = 'nyt_vocab.dat'
    data = pd.read_csv(data_path, sep='\n', header=None)
    vocab = pd.read_csv(vocab_path, sep='\n', header=None, names=['words'])

    N = 3012 #number of words
    M = 8447 #number of articles
    X = np.zeros((N, M)) #define the original data matrix

    #load data, every row is a single document
    #format "idx:cnt" with commas separating each unique word in the document
    for column in range(M):
        row = data.iloc[column].values[0].split(',')
        for item in row:
            index, count = map(int, item.split(':'))
            X[index-1][column] = count
    
    K = 25

    # Device memory allocation for input and output array(s)
    W = np.float32(np.random.uniform(1, 2, (N, K))) #initialize W and H to random values between 1 and 2
    H = np.float32(np.random.uniform(1, 2, (K, M)))

    #parallel implementation***************************************************************************
    #call NMF function
    H,W,time = test.NMF(np.float32(X), N, M, K, np.float32(W), np.float32(H))

    W = W / W.sum(axis=0).reshape(1,-1) #normalize the 25 categories
    W1 = W

    #find the 10 most used words for each category and print
    data = pd.DataFrame(index=range(10), columns=['Topic_%d' % i for i in range(1, 26)])
    for i in range(25):
        column = 'Topic_' + str(i+1)
        Wi = W[:, i]
        dt = pd.DataFrame(Wi, columns=['weight'])
        dt['words'] = vocab
        dt = dt.sort_values(by='weight', ascending=False)[:10].reset_index(drop=True)
        data[column] = dt['weight'].map(lambda x: ('%.4f')%x) + ' ' + dt['words']
    print(data)

    #numpy implementation******************************************************************************
    iteration = 100
    err = 1e-16 #a small error to prevent 0/0

    start = timer.time() #record start time

    for i in range(1, iteration+1):
        if i % 10 == 0:
            print('iteration %d' % i)
            
        Wt = W.T.astype(np.float32) #w transpose
        H = H * Wt.dot(X) / (Wt.dot(W).dot(H) + err) #update H

        Ht = H.T.astype(np.float32) #H transpose
        W = W * X.dot(Ht) / (W.dot(H).dot(Ht) + err) #update W

    end = timer.time() #record end time
    
    W = W / W.sum(axis=0).reshape(1,-1) #normalize the 25 categories
    
    #find the 10 most used words for each category and print
    data = pd.DataFrame(index=range(10), columns=['Topic_%d' % i for i in range(1, 26)])
    for i in range(25):
        column = 'Topic_' + str(i+1)
        Wi = W[:, i]
        dt = pd.DataFrame(Wi, columns=['weight'])
        dt['words'] = vocab
        dt = dt.sort_values(by='weight', ascending=False)[:10].reset_index(drop=True)
        data[column] = dt['weight'].map(lambda x: ('%.4f')%x) + ' ' + dt['words']
    print(data)

    if((W == W1).all()):
        print("W test case passed")

    print("python serial time:")
    print(end-start)
    print("cuda parallel time:")
    print(time)

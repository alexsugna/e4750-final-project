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

        kernelwrapper = str(func1) + str(func2);
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
    
    def NMF(self, X, N, M, K):

        ACols = a.shape[1] #get number of columns of input A
        ARows = a.shape[0] #get number of rows of input A
        
        # Get kernel function
        func_mul = self.mod.get_function("MatMul")
        func_tran = self.mod.get_function("MatTran")

        # Device memory allocation for input and output array(s)
        W = np.float32(np.random.uniform(1, 2, (N, K))) #initialize W and H to random values between 1 and 2
        H = np.float32(np.random.uniform(1, 2, (K, M)))
        
        X_d = gpuarray.to_gpu(X)
        W_d = gpuarray.to_gpu(W)
        H_d = gpuarray.to_gpu(H)

        # Record execution time and execute operation.

        block_dim = (32,32,1)
        grid_dim = (1,1,1)
        event = func(a_d, c_d, np.int32(ACols), np.int32(ARows), block=block_dim, grid=grid_dim)
        
        # Wait for the event to complete
        cuda.Context.synchronize()

        # Fetch result from device to host
        c = c_d.get()

        # Convert output array back to string

        return c #, time_


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
    '''
    #NMF test data set
    data_path = 'nyt_data.txt'
    vocab_path = 'nyt_vocab.dat'
    data = pd.read_csv(data_path, sep='\n', header=None)
    vocab = pd.read_csv(vocab_path, sep='\n', header=None, names=['words'])

    N = 3012 #number of words
    M = 8447 #number of articles
    X = np.zeros((N, M)) #define the original data matrix

    #load data, every row is a single document
    #format “idx:cnt” with commas separating each unique word in the document
    for column in range(M):
        row = data.iloc[column].values[0].split(',')
        for item in row:
            index, count = map(int, item.split(':'))
            X[index-1][column] = count
    
    K = 25
    result = test.NMF(np.float32(X), N, M, K)
    '''

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

        text_file = open("/home/sls2305/Desktop/e4750-final-project-main/kernels/RowColSum.cu", "r")
        func4 = text_file.read()
        text_file.close()

        kernelwrapper = str(func1) + str(func2) + str(func3) + str(func4);
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

    def ColSum(self, a):

        ACols = a.shape[1] #get number of columns of input A
        ARows = a.shape[0] #get number of rows of input A
        
        # Get kernel function
        func = self.mod.get_function("column_sum")
        func_col_div = self.mod.get_function("MatEleDivideCol") #divide col by sum

        # Device memory allocation for input and output array(s)
        c = np.zeros( ((1,ACols)), dtype=np.float32)
        
        a_d = gpuarray.to_gpu(a)
        c_d = gpuarray.to_gpu(c)

        # Record execution time and execute operation.
        block_dim, grid_dim = self.getGridDimention(1,ACols)
        
        event = func(a_d, c_d, np.int32(ARows), np.int32(ACols), block=block_dim, grid=grid_dim)
        
        # Wait for the event to complete
        cuda.Context.synchronize()

        block_dim, grid_dim = self.getGridDimention(ARows,ACols)
        event = func_col_div(a_d, c_d, np.int32(ARows), np.int32(ACols), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        # Fetch result from device to host
        a = a_d.get()

        # Convert output array back to string

        return a #, time_

    def RowSum(self, a):

        ACols = a.shape[1] #get number of columns of input A
        ARows = a.shape[0] #get number of rows of input A
        
        # Get kernel function
        func = self.mod.get_function("row_sum")
        func_row_div = self.mod.get_function("MatEleDivideRow") #divide row by sum

        # Device memory allocation for input and output array(s)
        c = np.zeros( ((ARows,1)), dtype=np.float32)
        
        a_d = gpuarray.to_gpu(a)
        c_d = gpuarray.to_gpu(c)

        # Record execution time and execute operation.
        block_dim, grid_dim = self.getGridDimention(ARows,1)
        
        event = func(a_d, c_d, np.int32(ARows), np.int32(ACols), block=block_dim, grid=grid_dim)
        
        # Wait for the event to complete
        cuda.Context.synchronize()

        block_dim, grid_dim = self.getGridDimention(ARows,ACols)
        event = func_row_div(a_d, c_d, np.int32(ARows), np.int32(ACols), block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()

        # Fetch result from device to host
        a = a_d.get()

        # Convert output array back to string

        return a #, time_
    
    def NMF(self, X, N, M, K, W, H):

        #ACols = a.shape[1] #get number of columns of input A
        #ARows = a.shape[0] #get number of rows of input A
        
        # Get kernel function
        func_mul = self.mod.get_function("MatMul") #matrix multiplication
        func_ele_mul = self.mod.get_function("MatEleMulInPlace") #matrix multiplication elementwise
        func_tran = self.mod.get_function("MatTran") #matrix transpose
        func_add = self.mod.get_function("MatEleAddInPlace") #matrix elementwise addition
        func_div = self.mod.get_function("MatEleDivideInPlace") #matrix elementwise division
        func_divC = self.mod.get_function("MatEleDivide") #gives output C matrix
        func_row_sum = self.mod.get_function("row_sum") #matrix elementwise division
        func_col_sum = self.mod.get_function("column_sum") #matrix elementwise division
        func_row_div = self.mod.get_function("MatEleDivideRow") #divide row by sum
        func_col_div = self.mod.get_function("MatEleDivideCol") #divide col by sum
        
        # Event objects to mark start and end points
        g_start = cuda.Event()
        g_end = cuda.Event()
        g_start.record()

        #define X, W, and H on gpu
        X_d = gpuarray.to_gpu(X)
        W_d = gpuarray.to_gpu(W)
        H_d = gpuarray.to_gpu(H)
        
        #itermediate steps for H update
        WH_d = gpuarray.zeros(((N,M)), dtype=np.float32)
        P_d = gpuarray.zeros(((N,M)), dtype=np.float32)
        Wt_d = gpuarray.zeros(((K,N)), dtype=np.float32)
        Wt_sum_d = gpuarray.zeros(((K,1)), dtype=np.float32) #sum rows
        WtP_d = gpuarray.zeros(((K,M)), dtype=np.float32)

        #define intermediate steps on gpu for W update
        Ht_d = gpuarray.zeros(((M,K)), dtype=np.float32)

        Ht_sum_d = gpuarray.zeros(((1,K)), dtype=np.float32) #sum cols

        PHt_d = gpuarray.zeros(((N,K)), dtype=np.float32)

        #get block dim and grid dim
        
        err = 1e-16 #a small error to prevent 0/0
        iteration = 100 #number of iterations

        # Record execution time and execute operation.
        for i in range(1, iteration+1):
            if i % 10 == 0:
                print('iteration %d' % i)
            
            #W (N,K)
            #H (K,M)
            #X (N,M)
            
            #UPDATE H *****************************************************************************
            #P = X / (W.dot(H)+eps) #intermediate step (purple matrix in notes)
            #Wt = W.T
            #Wt = Wt / Wt.sum(axis=1).reshape(-1, 1) #normalize rows
            #H = H * (Wt.dot(P))  #update H

            #W.dot(H)
            block_dim, grid_dim = self.getGridDimention(N,M)
            event = func_mul(W_d, H_d, WH_d, np.int32(N), np.int32(K), np.int32(K), np.int32(M), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #WH + err
            block_dim, grid_dim = self.getGridDimention(N,M)
            event = func_add(WH_d, np.float32(err), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #X / WH (saved as P_d)
            block_dim, grid_dim = self.getGridDimention(N,M)
            event = func_divC(X_d, WH_d, P_d, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #Wt = W.T
            block_dim, grid_dim = self.getGridDimention(N,K)
            event = func_tran(W_d, Wt_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            

            #Wt.sum(axis=1).reshape(-1, 1) #sum rows
            block_dim, grid_dim = self.getGridDimention(K,1)
            event = func_row_sum(Wt_d, Wt_sum_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize() 

            #Wt = Wt / Wt_sum_d #elementwise
            block_dim, grid_dim = self.getGridDimention(K,N)
            event = func_row_div(Wt_d, Wt_sum_d, np.int32(K), np.int32(N), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #Wt.dot(P)
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_mul(Wt_d, P_d, WtP_d, np.int32(K), np.int32(N), np.int32(N), np.int32(M), np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            
            #H .* WtP elementwise
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_ele_mul(H_d, WtP_d, np.int32(K), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #W UPDATE************************************************************************************
            #P = X / (W.dot(H)+eps)
            #Ht = H.T
            #Ht = Ht / Ht.sum(axis=0).reshape(1, -1) #normalize columns
            #W = W * (P.dot(Ht))  #update W
    
            #W.dot(H)
            block_dim, grid_dim = self.getGridDimention(N,M)
            event = func_mul(W_d, H_d, WH_d, np.int32(N), np.int32(K), np.int32(K), np.int32(M), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()
            
            

            #WH + err
            block_dim, grid_dim = self.getGridDimention(N,M)
            event = func_add(WH_d, np.float32(err), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #X / WH (saved as P_d)
            block_dim, grid_dim = self.getGridDimention(N,M)
            event = func_divC(X_d, WH_d, P_d, np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #Ht = H.T #H transpose
            block_dim, grid_dim = self.getGridDimention(K,M)
            event = func_tran(H_d, Ht_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #Ht.sum(axis=0).reshape(1, -1) #sum columns
            block_dim, grid_dim = self.getGridDimention(1,K)
            event = func_col_sum(Ht_d, Ht_sum_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            Ht = Ht_d.get()
            
            #Ht = Ht / Ht_sum_d #elementwise
            block_dim, grid_dim = self.getGridDimention(M,K)
            event = func_col_div(Ht_d, Ht_sum_d, np.int32(M), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            #P.dot(Ht)
            block_dim, grid_dim = self.getGridDimention(N,K)
            event = func_mul(P_d, Ht_d, PHt_d, np.int32(N), np.int32(M), np.int32(M), np.int32(K), np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()

            W = W_d.get()

            #W .* PHt elementwise
            block_dim, grid_dim = self.getGridDimention(N,K)
            event = func_ele_mul(W_d, PHt_d, np.int32(N), np.int32(K), block=block_dim, grid=grid_dim)
            cuda.Context.synchronize()   
            
            
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
    
    a = np.random.rand(8447, 3012)
    b = np.random.randint(5, size=(5, 5))
    print("a:",a)
    print("b:",b)
    '''
    #multiplication test case
    result = test.MatMul(np.float32(a),np.float32(b))
    #print("multiply result:",result)
    serial_mul = a.dot(b) #numpy multiply test case
    #print("serial multiply result:",serial_mul)
    if((result == serial_mul).all()):
        print("multiply test case passed")
    
    #transpose test case
    result = test.MatTran(np.float32(a))
    #print("transpose result:",result)
    serial_tran = np.float32(a).T
    if((result == serial_tran).all()):
        print("transpose test case passed")
    
    
    #col_sum test case
    result = test.ColSum(np.float32(a))
    serial_column = a / a.sum(axis=0).reshape(1, -1)
    print(result)
    print(serial_column)
    if((result == serial_column).all()):
        print("column sum test case passed")
    
    #row_sum test case
    result = test.RowSum(np.float32(a))
    serial_row = a / a.sum(axis=1).reshape(-1, 1)
    print(result)
    print(serial_row)
    if((result == serial_row).all()):
        print("row sum test case passed")
    
    '''
    #NMF test data set
    data_path = 'nyt_data.txt'
    vocab_path = 'nyt_vocab.dat'
    data = pd.read_csv(data_path, sep='\n', header=None)
    vocab = pd.read_csv(vocab_path, sep='\n', header=None, names=['words'])

    N = 3012 #number of words
    M = 8447 #number of articles
    X = np.float32(np.zeros((N, M))) #define the original data matrix

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
    H_out,W_out,time = test.NMF(np.float32(X), N, M, K, np.float32(W), np.float32(H))

    W_out = W_out / W_out.sum(axis=0).reshape(1,-1) #normalize the 25 categories

    #find the 10 most used words for each category and print
    data = pd.DataFrame(index=range(10), columns=['Topic_%d' % i for i in range(1, 26)])
    for i in range(25):
        column = 'Topic_' + str(i+1)
        Wi = W_out[:, i]
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
            
        P = X / (W.dot(H)+err) #intermediate step (purple matrix in notes)
        Wt = W.T
        Wt = Wt / Wt.sum(axis=1).reshape(-1, 1) #normalize rows
        H = H * (Wt.dot(P))  #update H
        P = X / (W.dot(H)+err)
        Ht = H.T
        Ht = Ht / Ht.sum(axis=0).reshape(1, -1) #normalize columns
        W = W * (P.dot(Ht))  #update W

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

    if((W == W_out).all()):
        print("W test case passed")

    print("python serial time:")
    print(end-start)
    print("cuda parallel time:")
    print(time)

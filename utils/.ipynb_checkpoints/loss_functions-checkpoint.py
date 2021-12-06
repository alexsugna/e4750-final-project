"""
Parallel implementations of loss functions for NMF
"""
import pycuda.gpuarray as gpuarray
from utils.context import Context


BLOCK_SIZE = 32
context = Context(BLOCK_SIZE)

matrix_multiplication_kernel_path = './kernels/MatrixMultiplication.cu'
matrix_transpose_kernel_path = './kernels/MatrixTranspose.cu'
element_multiplication_kernel_path = './kernels/ElementWise.cu'

kernel_paths = [matrix_multiplication_kernel_path, 
                matrix_transpose_kernel_path,
                element_multiplication_kernel_path]

source_module = context.getSourceModule(kernel_paths, multiple_kernels=True)

def euclidean(X, W, H):
    """
    sum((X - WH)^2)
    """
    
    # compute WH
    W = W.astype(np.float32)
    H = H.astype(np.float32)
    
    WH = np.zeros((W.shape[0], H.shape[1])).astype(np.float32)
    
    W_d = gpuarray.to_gpu(W)
    H_d = gpuarray.to_gpu(H)
    WH_d = gpuarray.to_gpu(WH)
    
    block = context.block_dims
    grid = context.grid_dims(max([W.shape[0], H.shape[1]]))
    
    matrix_multiplication(a_d, b_d, c_d, np.int32(a.shape[0]), 
                      np.int32(a.shape[1]), np.int32(b.shape[0]), 
                      np.int32(b.shape[1]), np.int32(c.shape[0]), np.int32(c.shape[1]),
                      block=block, grid=grid)
    
    
    
    
    
    
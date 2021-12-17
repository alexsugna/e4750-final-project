import numpy as np
import pycuda.gpuarray as gpuarray

from .context import Context

BLOCK_SIZE = 32
context = Context(BLOCK_SIZE)

matrix_multiplication_kernel_path = './kernels/MatrixMultiplication.cu'
matrix_transpose_kernel_path = './kernels/MatrixTranspose.cu'
element_multiplication_kernel_path = './kernels/ElementWise.cu'
matrix_sum_path = './kernels/MatrixSum.cu'

kernel_paths = [matrix_multiplication_kernel_path, 
                matrix_transpose_kernel_path,
                element_multiplication_kernel_path,
                matrix_sum_path]

source_module = context.getSourceModule(kernel_paths, multiple_kernels=True)


def matrix_multiplication(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    
    ab = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)
    
    a_d = gpuarray.to_gpu(a)
    b_d = gpuarray.to_gpu(b)
    ab_d = gpuarray.to_gpu(ab)
    
    block = context.block_dims
    grid = context.grid_dims(max([a.shape[0], b.shape[1]]))
    
    matrix_mult = source_module.get_function('MatMul')
    
    matrix_mult(a_d, b_d, ab_d, np.int32(a.shape[0]), 
                      np.int32(a.shape[1]), np.int32(b.shape[0]), 
                      np.int32(b.shape[1]), np.int32(ab.shape[0]),
                      np.int32(ab.shape[1]), block=block, grid=grid)
    
    return ab_d.get()

def matrix_subtract(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c = np.zeros(a.shape).astype(np.float32)
    
    a_d = gpuarray.to_gpu(a)
    b_d = gpuarray.to_gpu(b)
    c_d = gpuarray.to_gpu(c)
    
    block = context.block_dims
    grid = context.grid_dims(max(a.shape))
    
    matrix_subtraction = source_module.get_function('MatEleSubtract')
    
    matrix_subtraction(a_d, b_d, c_d, np.int32(a.shape[0]), np.int32(a.shape[1]), block=block, grid=grid)
    
    return c_d.get()


def matrix_square(a):
    a = a.astype(np.float32)
    c = np.zeros(a.shape).astype(np.float32)
    
    a_d = gpuarray.to_gpu(a)
    c_d = gpuarray.to_gpu(c)
    
    block = context.block_dims
    grid = context.grid_dims(max(a.shape))
    
    matrix_sq = source_module.get_function('MatEleSquare')
    
    matrix_sq(a_d, c_d, np.int32(a.shape[0]), np.int32(a.shape[1]), block=block, grid=grid)
    
    return c_d.get()


def matrix_sum(input_array):
    
    input_array = input_array.astype(np.float32)
    length = len(input_array.flatten())
    
    if length > BLOCK_SIZE:  
        # call hierarchical prefix sum
        output = prefix_sum_multiple_kernel(input_array)
        
    else:
        # call single block prefix sum
        output = step_2(input_array)
    
    return output[-1]
    
    
def prefix_sum_multiple_kernel(input_array):
    """
    For input arrays that are larger than BLOCK_SIZE, we need to coordinate
    between thread blocks. This is only possible by making multiple kernel calls.
    This function coordinates the kernel calls for large input. 
    """        
    if len(input_array.shape) > 1: # flatten 2D input
        input_array = input_array.flatten()

    partially_scanned_input, S = step_1(input_array) # perform blockwise scan

    """
    If intermediate scan result S is larger than block size, we need to use more than one block 
    to perform the scan across S, thus we make a recursive call to prefix_sum_multiple_kernel()
    to scan input larger than block size. If S < block size, scan on a single block.
    """
    if len(S) > BLOCK_SIZE:
        S_scanned = prefix_sum_multiple_kernel(S)

    else:
        S_scanned = step_2(S)

    # execute step 3, distribute the results of the scan of S to the partially scanned input
    output_array = step_3(S_scanned, partially_scanned_input)

    return output_array
    
    
def step_1(input_array):
    """
    Step 1 of multi-kernel scan. This function takes the input array, 
    splits it into blocks, performs an individual scan on each block,
    stores the terminal value of the individual scans in an array S, 
    then returns the partially scanned input array and the terminal 
    values S.
    """
    input_array = input_array.astype(np.float32)
    length = len(input_array)
    output_array = np.zeros(input_array.shape).astype(np.float32) # define input/output arrays

    s_length = int(length/BLOCK_SIZE) # get length of S
    s_array = np.zeros(s_length).astype(np.float32) # initialize S

    input_array_d = gpuarray.to_gpu(input_array) # allocate memory on GPU
    output_array_d = gpuarray.to_gpu(output_array)

    s_array_d = gpuarray.to_gpu(s_array)

    prefixSum = source_module.get_function('workInefficientPrefixSum')

    block = (BLOCK_SIZE, 1, 1)
    grid = (int(np.ceil(length / BLOCK_SIZE)), 1, 1)

    prefixSum(input_array_d, output_array_d, s_array_d, # make kernel call
              np.int32(length), block=block, grid=grid)

    output_array = output_array_d.get() # transfer result back to CPU
    s_array = s_array_d.get()

    return output_array, s_array # return partially scanned input and S
    
    
def step_2(s_array):
    """
    Perform a scan on the terminal values of the input scans.

    Same process as step 1, but we only care about the result of the 
    scan of S.
    """
    s_length = len(s_array)
    s_array_out = np.zeros(s_array.shape).astype(np.float32)
    s_placeholder = np.zeros(s_length).astype(np.float32)

    s_array_in_d = gpuarray.to_gpu(s_array)
    s_array_out_d = gpuarray.to_gpu(s_array_out)
    s_placeholder_d = gpuarray.to_gpu(s_placeholder)

    
    prefixSum = source_module.get_function('workInefficientPrefixSum')
        

    block = (BLOCK_SIZE, 1, 1)
    grid = (int(np.ceil(s_length / BLOCK_SIZE)), 1, 1)

    prefixSum(s_array_in_d, s_array_out_d, s_placeholder_d, np.int32(s_length), 
              block=block, grid=grid)

    S_scanned = s_array_out_d.get()

    return S_scanned

def step_3(S_scanned, partially_scanned_input):
    """
    This function takes the scanned S and distributes the results to
    the intermediately scanned input array.
    """
    S_scanned_d = gpuarray.to_gpu(S_scanned) # allocate memory on GPU
    partially_scanned_input_d = gpuarray.to_gpu(partially_scanned_input)

    length = len(partially_scanned_input) # get length of input

    block = (BLOCK_SIZE, 1, 1) # define block and grid dimensions
    grid = (int(np.ceil(length / BLOCK_SIZE)), 1, 1)

    distributeScanResults = source_module.get_function('distributeScanResults') # get kernel function

    # call kernel function
    distributeScanResults(partially_scanned_input_d, S_scanned_d, np.int32(length), 
                          block=block, grid=grid)

    output_array = partially_scanned_input_d.get() # transfer result to CPU

    return output_array
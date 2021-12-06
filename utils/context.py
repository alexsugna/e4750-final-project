from pycuda.compiler import SourceModule
from pycuda import autoinit

import numpy as np


class Context:
    def __init__(self, block_size):
        
        self.block_size = block_size
        
        self.block_dims = (self.block_size, self.block_size, 1) # define block and grid dimensions
        self.grid_dims = lambda length: (int(np.ceil(length / self.block_size)), int(np.ceil(length / self.block_size)), 1)
        
    
    def getSourceModule(self, kernel_path, multiple_kernels=False):
        
        if multiple_kernels:
            try:
                is_iterable = iter(kernel_path)
                kernelwrapper = ''
                for kernel in kernel_path:
                    kernelwrapper += open(kernel).read()
                return SourceModule(kernelwrapper)
                
            except TypeError as te:
                print(kernel_path, 'is not iterable. Set multiple_kernels=False or define kernel_path as list of paths.')
        else:
            kernelwrapper = open(kernel_path).read() # read CUDA kernels from .cu file

        return SourceModule(kernelwrapper)

    

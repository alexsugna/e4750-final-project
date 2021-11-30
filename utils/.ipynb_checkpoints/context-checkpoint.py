from pycuda.compiler import SourceModule
from pycuda import autoinit

import numpy as np


class Context:
    def __init__(self, block_size):
        
        self.block_size = block_size
        
        self.block_dims = (self.block_size, self.block_size, 1) # define block and grid dimensions
        self.grid_dims = lambda length: (int(np.ceil(length / self.block_size)), int(np.ceil(length / self.block_size)), 1)
        
     
    def getSourceModule(self, kernel_path):
        
        kernelwrapper = open(kernel_path).read() # read CUDA kernels from .cu file

        return SourceModule(kernelwrapper)

    

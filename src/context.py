"""
This file contains the class "Context" which initializes the PyCUDA environment
and
"""
from pycuda.compiler import SourceModule
from pycuda import autoinit
import numpy as np

class Context:
    def __init__(self, block_size):
        """
        Initialize the CUDA context.

        params:
            block_size (int): defines the block size for parallel computation
        """
        self.block_size = block_size

        self.block_dims = (self.block_size, self.block_size, 1) # define block and grid dimensions
        # if x and y are same size use grid_dims:
        self.grid_dims = lambda length: (int(np.ceil(length / self.block_size)), int(np.ceil(length / self.block_size)), 1)
        # if x and y are not the same size use grid_dims2d:
        self.grid_dims2d = lambda x, y : (int(np.ceil(y / self.block_size)), int(np.ceil(x / self.block_size)), 1)

    def getSourceModule(self, kernel_path, multiple_kernels=False):
        """
        Load the CUDA kernels. Can load kernels from a single file or from
        multiple files.

        params:
            kernel_path (str or list of str): the path to the .cu CUDA kernel file
                                              or list of paths. If list of paths then
                                              multiple_kernels must be set to True.

            multiple_kernels (bool): specifies if kernel_path contains one path
                                     or list of paths.

        returns:
            Instance of pycuda.compiler.SourceModule with user parallel kernels loaded.
        """
        if multiple_kernels:
            try:
                is_iterable = iter(kernel_path)
                kernelwrapper = ''
                for kernel in kernel_path: # if kernel_path is a list, iteratively read files
                    kernelwrapper += open(kernel).read() # concatenate files as strings
                return SourceModule(kernelwrapper) # load concatenated kernel code into source module

            except TypeError:
                raise Exception('{} is not iterable. Set multiple_kernels=False or define kernel_path as list of paths.'.format(kernel_path))
        else:
            kernelwrapper = open(kernel_path).read() # read CUDA kernels from .cu file

        return SourceModule(kernelwrapper) # return source module with kernels loaded

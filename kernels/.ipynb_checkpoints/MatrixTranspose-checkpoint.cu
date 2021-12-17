// adapted from https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu

#define BLOCK_SIZE 32

__global__ void MatTran(float *idata, float *odata, int out_width, int out_height)
{
	__shared__ float block[BLOCK_SIZE][BLOCK_SIZE+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	if((xIndex < out_width) && (yIndex < out_height))
	{
		unsigned int index_in = yIndex * out_width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
	if((xIndex < out_height) && (yIndex < out_width))
	{
		unsigned int index_out = yIndex * out_height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}



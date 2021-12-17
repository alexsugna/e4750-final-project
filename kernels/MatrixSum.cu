
#define SECTION_SIZE 32

__global__ void workInefficientPrefixSum(float *input, float *output, float *S, int length){
    
    __shared__ float temp[SECTION_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < length){
        temp[threadIdx.x] = input[i];
    }
    
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if (threadIdx.x >= stride){
            temp[threadIdx.x] += temp[threadIdx.x - stride];
        }
    }
    __syncthreads();
    
    if(threadIdx.x == blockDim.x - 1){
        S[blockIdx.x] = temp[SECTION_SIZE - 1];
    }
    
    output[i] = temp[threadIdx.x];
}

__global__ void distributeScanResults(float *output, float *S, int length){
    
    if(blockIdx.x > 0){
    
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(i < length){
            output[i] += S[blockIdx.x - 1];
        }
    }
}

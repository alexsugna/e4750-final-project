// element-wise matrix multiplication and add

#define BLOCK_SIZE 32

__global__ void MatEleMul(float* A, float* B, float *C, int width, int height){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x < width && y < height){
        C[y * width + x] = A[y * width + x] * B[y * width + x];
    }
}

__global__ void MatEleAdd(float* A, float* B, float *C, int width, int height){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x < width && y < height){
        C[y * width + x] = A[y * width + x] + B[y * width + x];
    }
}
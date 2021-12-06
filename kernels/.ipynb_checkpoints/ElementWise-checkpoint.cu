// element-wise matrix multiplication and add

__global__ void MatEleMul(float* A, float* B, float *C, int width, int height){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x < width && y < height){
        int index = y * width + x;
        C[index] = A[index] * B[index];
    }
}

__global__ void MatEleAdd(float* A, float* B, float *C, int width, int height){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x < width && y < height){
        int index = y * width + x;
        C[index] = A[index] + B[index];
    }
}
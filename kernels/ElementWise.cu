// element-wise matrix operations

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

__global__ void MatEleSubtract(float* A, float* B, float *C, int width, int height){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x < width && y < height){
        int index = y * width + x;
        C[index] = A[index] - B[index];
    }
}

__global__ void MatEleSquare(float* A, float *C, int width, int height){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x < width && y < height){
        int index = y * width + x;
        C[index] = A[index] * A[index];
    }
}

__global__ void MatEleDivide(float* A, float* B, float *C, int width, int height){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x < width && y < height){
        int index = y * width + x;
        C[index] = A[index] / B[index];
    }
}
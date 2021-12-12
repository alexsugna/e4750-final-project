//column wise sum operation

__global__ void column_sum(float* A,  float* result, int height, int width){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    while (idx < width){
        float my_result = 0;
        for (int i = 0; i < height; i++){
            my_result += A[(i*width)+idx];
        }
        result[idx] = my_result;
        
        idx += gridDim.x * blockDim.x;
    }
}

__global__ void row_sum(float* A, float* result, int height, int width){

    int idy = threadIdx.y + blockDim.y*blockIdx.y;
    while (idy < height){
        float my_result = 0;
        for (int i = 0; i < width; i++){
            my_result += A[(i + width*idy)];
        }
        result[idy] = my_result;
        idy += gridDim.y * blockDim.y;
    }
}

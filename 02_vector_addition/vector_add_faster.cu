/*
STEPS 
1. Allocate host memory and initialized host data e.g. malloc
2. Allocate device memory e.g cudaMalloc
3. Transfer input data from host to device memory e.g cudaMemcpy
4. Execute kernels
5. Transfer output from device memory to host
6. Free Host & CUDA memory e.g. free & cudaFree
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index<n)
        out[index] = a[index] + b[index];
}

int main(){
    float *a, *b, *out; 
    float *d_a, *d_b, *d_out;

    // 1. Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // 1. Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // 2. Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // 3. Transfer input data from host to device
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // 4. Kernel launch
    int block_size = 256; // 256 threads in each block
    int grid_size = ((N + block_size) / block_size); // i.e. 39063 blocks // add blocksize to avoid 1 missing integer index. e.g. N = 20, block_size = 7 then grid_size = 20/7 = 2 and it will access threads only upto 13 index
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);  // no. of threads  = N

    // 5. Transfer output from device memory to host
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("PASSED\n");

    // 6. Free cuda memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // 6. Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}


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
    /*
    For the k-th thread, the loop starts from k-th element and iterates 
    through the array with a loop stride of 256. For example, in the 0-th 
    iteration, the k-th thread computes the addition of k-th element. In the 
    next iteration, the k-th thread computes the addition of (k+256)-th 
    element, and so on. Following figure shows an illustration of the idea.
    */
    int index = threadIdx.x; // thread identifier inside block. values from 0-255
    int stride = blockDim.x; // number of threads in a block i.e. 256
    /*
    1. Compilation: there will be 256 copies of this function for each thread
    2. Initialize: i holds value of thread idx i.e. 0 to 255 (each thread has unique identifier)
    3. Increment: each i value then increamented by 255 (block dimension) hence loop will run for N/256 itreation
    */    
    for(int i = index; i < n; i+=stride){
        out[i] = a[i] + b[i];
    }
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
    vector_add<<<1,256>>>(d_out, d_a, d_b, N);      

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


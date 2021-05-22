/*
STEPS 
1. Allocate host memory and initialized host data e.g. malloc
2. Allocate device memory e.g cudaMalloc
3. Transfer input data from host to device memory e.g cudaMemcpy
4. Execute kernels
5. Transfer output from device memory to host
6. Free cuda memory e.g. cudaFree
*/


#include<stdio.h>
#include<stdlib.h>

#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
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
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);      //use only one thread

    // 5. Transfer output from device memory to host
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // 6. Free cuda memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}


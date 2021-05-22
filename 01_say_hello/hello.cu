#include<stdio.h>

// called kernel in GPU ( in CPU its called function)
__global__ void hello() {           //__global__ is specifier indicating function runs in GPU (aka device)
    printf("Hello CUDA\n");
}

int main() {
// execute kernal 
    hello <<<1,1>>>();              //<<<M,T>>> M - #ThreadBlock & T - #Threads
    return 0;
}

// To compile: $ nvcc hello.cu -o hello_cu
// To execute: $ ./hello_cu
// To inspect execution time: $ nvprof ./hello_cu

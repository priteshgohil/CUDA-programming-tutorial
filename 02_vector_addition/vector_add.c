// source https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    clock_t begin = clock();
    vector_add(out, a, b, N);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("vector_add execution time in Seconds: %f \n", time_spent);
}


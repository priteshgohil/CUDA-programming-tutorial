# Introduction
The CUDA C program is similar to C program. CUDA is a platform and programming model for CUDA-enabled GPUs. CUDA provides language extention for C or C++ and API for the programming and manage the usage of GPU for your application. 

CUDA programming uses CPU (host) as well as GPU (device) for computing. CPU are generally used for the serial workflow. However, GPU is useful to run parallel offload computation. Both CPU and GPU have their own memory.

## Running first small program on GPU
If you compare the hello.c and hello.cu, they both look identical except for the following changes
- `__global__` specifier indicate that kernal (function) runs on a device (GPU)
- `<<<1,1>>>` indicate the run configuration. More specifically, number of threads to use for the kernal launch. 
 
Configuration is dicussed in more detail in next session.

### Build
The program can be built using cuda compiler. In this nvidia cuda compiler is used,
`nvcc hello.cu -o hello_cu`

## Run
Though running cuda program will not print on the console
`./hello_cu`

## Profiling GPU time
Profiling will give information about the time it took to run the application
`nvprof ./hello_cu`
If you do not have permission for profiling, you can use sudo 
`which nvprof`
`sudo NVPROF_FULL_PATH ./hello_cu`

```==9819== Profiling application: ./hello_cu
==9819== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  254.45us         1  254.45us  254.45us  254.45us  hello(void)
      API calls:   99.96%  328.12ms         1  328.12ms  328.12ms  328.12ms  cudaLaunchKernel
                    0.03%  100.74us        97  1.0380us     416ns  36.255us  cuDeviceGetAttribute
                    0.00%  15.008us         1  15.008us  15.008us  15.008us  cuDeviceTotalMem
                    0.00%  5.3120us         3  1.7700us     992ns  2.6240us  cuDeviceGetCount
                    0.00%  4.1280us         2  2.0640us     960ns  3.1680us  cuDeviceGet
                    0.00%  2.4320us         1  2.4320us  2.4320us  2.4320us  cuDeviceGetName
                    0.00%     512ns         1     512ns     512ns     512ns  cuDeviceGetUuid
```

## Conclusion
This tutorial should help you to write your first CUDA kernel, understand the basics of CUDA and identify CUDA kernal.

## References
- https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/

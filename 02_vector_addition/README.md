# Introduction
This tutorial aims to compute the addition of two vector `a`, `b` and store result in `out`. `vector_add.c` is the C program that runs on CPU and all the files with `.cu` extensions contains the code to perform vector addition on GPU.

## Vector addition on GPU
![two vector addition using C](../images/vector_add.png "Add vector a and b")

### Build
- `vector_add.cu` uses only single thread on GPU to perform vector addition. 
- `vector_add_fast.cu` uses 256 threads on GPU to perform vector addition. 
- `vector_add_faster.cu` uses 39063 blocks and 256 threads (i.e. number of threads are equal to the size of vector N) on GPU to perform vector addition. 

`gcc vector_add.c -o vector_add_c`
`gcc vector_add.cu -o vector_add_cu`
`gcc vector_add_fast.cu -o vector_add_fast_cu`
`gcc vector_add_faster.cu -o vector_add_faster_cu`


## Run
`./vector_add_c`
`./vector_add_cu`

## Profiling GPU time
Profiling will give information about the time it took to run the application
`nvprof ./vector_add_cu`

```
==13239== Profiling application: ./vector_add_cu
==13239== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.45%  1.78038s         1  1.78038s  1.78038s  1.78038s  vector_add(float*, float*, float*, int)
                    1.87%  34.566ms         1  34.566ms  34.566ms  34.566ms  [CUDA memcpy DtoH]
                    1.68%  30.996ms         2  15.498ms  15.431ms  15.564ms  [CUDA memcpy HtoD]
      API calls:   84.20%  1.84737s         3  615.79ms  15.850ms  1.81549s  cudaMemcpy
                   15.62%  342.66ms         3  114.22ms  853.37us  336.15ms  cudaMalloc
                    0.16%  3.6000ms         3  1.2000ms  1.1673ms  1.2617ms  cudaFree
                    0.01%  142.75us        97  1.4710us     736ns  40.640us  cuDeviceGetAttribute
                    0.01%  124.83us         1  124.83us  124.83us  124.83us  cudaLaunchKernel
                    0.00%  13.024us         1  13.024us  13.024us  13.024us  cuDeviceTotalMem
                    0.00%  6.0800us         3  2.0260us     928ns  2.7520us  cuDeviceGetCount
                    0.00%  3.2640us         2  1.6320us  1.3760us  1.8880us  cuDeviceGet
                    0.00%  2.1760us         1  2.1760us  2.1760us  2.1760us  cuDeviceGetName
                    0.00%  1.0880us         1  1.0880us  1.0880us  1.0880us  cuDeviceGetUuid

```

`nvprof ./vector_add_fast_cu`
```
==12827== Profiling application: ./vector_add_fast_cu
==12827== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.35%  44.741ms         1  44.741ms  44.741ms  44.741ms  vector_add(float*, float*, float*, int)
                   31.44%  33.215ms         2  16.607ms  15.448ms  17.767ms  [CUDA memcpy HtoD]
                   26.21%  27.689ms         1  27.689ms  27.689ms  27.689ms  [CUDA memcpy DtoH]
      API calls:   69.32%  253.00ms         3  84.334ms  756.13us  251.46ms  cudaMalloc
                   29.56%  107.89ms         3  35.964ms  15.894ms  73.755ms  cudaMemcpy
                    1.06%  3.8706ms         3  1.2902ms  1.2481ms  1.3609ms  cudaFree
                    0.03%  114.14us         1  114.14us  114.14us  114.14us  cudaLaunchKernel
                    0.03%  97.600us        97  1.0060us     448ns  30.368us  cuDeviceGetAttribute
                    0.00%  11.040us         1  11.040us  11.040us  11.040us  cuDeviceTotalMem
                    0.00%  6.8800us         3  2.2930us  1.0240us  3.3600us  cuDeviceGetCount
                    0.00%  2.4320us         2  1.2160us     960ns  1.4720us  cuDeviceGet
                    0.00%  2.1120us         1  2.1120us  2.1120us  2.1120us  cuDeviceGetName
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetUuid
```

`nvprof ./vector_add_faster_cu`
```
==14108== Profiling application: ./vector_add_faster_cu
==14108== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   48.63%  30.792ms         2  15.396ms  15.366ms  15.426ms  [CUDA memcpy HtoD]
                   41.21%  26.091ms         1  26.091ms  26.091ms  26.091ms  [CUDA memcpy DtoH]
                   10.16%  6.4351ms         1  6.4351ms  6.4351ms  6.4351ms  vector_add(float*, float*, float*, int)
      API calls:   83.18%  344.02ms         3  114.67ms  658.11us  342.60ms  cudaMalloc
                   15.80%  65.342ms         3  21.781ms  15.780ms  33.650ms  cudaMemcpy
                    0.96%  3.9874ms         3  1.3291ms  1.3087ms  1.3648ms  cudaFree
                    0.03%  114.11us         1  114.11us  114.11us  114.11us  cudaLaunchKernel
                    0.03%  105.44us        97  1.0870us     448ns  35.072us  cuDeviceGetAttribute
                    0.00%  11.072us         1  11.072us  11.072us  11.072us  cuDeviceTotalMem
                    0.00%  5.5040us         3  1.8340us     992ns  2.6560us  cuDeviceGetCount
                    0.00%  2.9120us         2  1.4560us  1.1840us  1.7280us  cuDeviceGet
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceGetName
                    0.00%     704ns         1     704ns     704ns     704ns  cuDeviceGetUuid
```

## Conclusion
vector_add() time for different methods

| Method                  | Execution time (in mSec) | Speedup |
|-------------------------|--------------------------|---------|
| CPU                     | 88                       | -       |
| 1 block, 1 thread       | 1780                     | 1.00x   |
| 1 block, 256 thread     | 45                       | 39.55x  |
| 39063 block, 256 thread | 15                       | 118.66x |

## References
- https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/

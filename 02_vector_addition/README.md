# Introduction
explain about threads and blocks w/o diagram

## Vector addition on GPU


### Build
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

## Conclusion


## References
- https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/

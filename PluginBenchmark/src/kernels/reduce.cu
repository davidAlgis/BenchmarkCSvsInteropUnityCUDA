/**
 * @file reduce.cu
 * @brief Implementation of reduce.cuh
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */

#include "cuda_include.h"
#include "kernels/reduce.cuh"
#include <cub/cub.cuh>

int preAllocationReduce(float *d_array, float *d_result, void *d_tempStorage,
                        size_t tempStorageBytes, int arraySize)
{
    // pre allocation for cub::DeviceReduce::Sum see dedicated documentation
    // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html
    d_tempStorage = nullptr;
    tempStorageBytes = 0;
    CUDA_CHECK_RETURN(cub::DeviceReduce::Sum(d_tempStorage, tempStorageBytes,
                                             d_array, d_result, arraySize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_tempStorage, tempStorageBytes));
    return 0;
}

int reduce(float *d_array, float *d_result, void *d_tempStorage,
           size_t tempStorageBytes, int arraySize)
{
    // Sum the elements of the array
    CUDA_CHECK_RETURN(cub::DeviceReduce::Sum(d_tempStorage, tempStorageBytes,
                                             d_array, d_result, arraySize));
    return 0;
}
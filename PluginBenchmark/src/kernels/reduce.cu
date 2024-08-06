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

int preAllocationReduce(float *d_array, float *d_result, void **d_tempStorage,
                        size_t &tempStorageBytes, int arraySize)
{
    // pre allocation for cub::DeviceReduce::Sum see dedicated documentation
    // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html
    *d_tempStorage = nullptr;
    tempStorageBytes = 0;
    CUDA_CHECK_RETURN(cub::DeviceReduce::Sum(*d_tempStorage, tempStorageBytes,
                                             d_array, d_result, arraySize));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaMalloc(d_tempStorage, tempStorageBytes));
    if (tempStorageBytes == 0 || *d_tempStorage == nullptr)
    {
        Log::log().debugLogError(
            ("Failed to initialize temp storage for reduce in "
             "preAllocationReduce method. tempStorageBytes = " +
             std::to_string(tempStorageBytes) +
             " and arraySize = " + std::to_string(arraySize))
                .c_str());
        return -1;
    }
    return 0;
}

int reduce(float *d_array, float *d_result, void *d_tempStorage,
           size_t tempStorageBytes, int arraySize)
{
    if (d_tempStorage != nullptr)
    {
        // Sum the elements of the array

        auto ret = cub::DeviceReduce::Sum(d_tempStorage, tempStorageBytes,
                                          d_array, d_result, arraySize);
        if (ret != cudaSuccess)
        {
            Log::log().debugLogError(
                ("There has been an error while applying the reduce. "
                 "TempStorageBytes = " +
                 std::to_string(tempStorageBytes) +
                 " and arraySize = " + std::to_string(arraySize))
                    .c_str());
            return cudaAssert(ret, __FILE__, __LINE__);
        }
    }
    else
    {
        Log::log().debugLogError(
            "d_tempStorage has not been initialized with preAllocationReduce. "
            "Please initialize it before calling reduce.");
        return -1;
    }
    return 0;
}
/**
 * @file vector_add.cu
 * @brief Implementation of the kernel vectorAdd and \ref vector_add.cuh
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#include "cuda_runtime.h"
#include "kernels/vector_add.cuh"
#include "stdint.h"

__global__ void vectorAdd(const float *array1, const float *array2,
                          float *arrayResult, int arraySize)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < arraySize)
    {
        arrayResult[index] = array1[index] + array2[index];
    }
}

void kernelCallerVectorAdd(const float *array1, const float *array2,
                           float *arrayResult, int arraySize)
{
    // Calculate the block and grid sizes
    const std::uint32_t blockSize = 1024u;
    const std::uint32_t gridSize =
        (static_cast<std::uint32_t>(arraySize) + blockSize - 1u) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(array1, array2, arrayResult, arraySize);
}

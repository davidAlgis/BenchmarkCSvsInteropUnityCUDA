/**
 * @file vector_add.cuh
 * @brief Contains the function to call the kernel related to the \ref
 * ActionVectorAdd Which is simply the sum of 2 vectors
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#pragma once

/**
 * @brief      Add two array in a another one
 *
 * @param[in]  array1       The array 1
 * @param[in]  array2       The array 2
 * @param      arrayResult  The array result which is equal to the sum of array
 * 1 and 2
 * @param[in]  arraySize    The array size
 */
void kernelCallerWriteBuffer(const float *array1, const float *array2,
                             float *arrayResult, int arraySize);
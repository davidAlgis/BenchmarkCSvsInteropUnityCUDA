/**
 * @file reduce.cuh
 * @brief Contains the function to call reduce
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#pragma once

int preAllocationReduce(float *d_array, float *d_result, void **d_tempStorage,
                        size_t tempStorageBytes, int arraySize);

int reduce(float *d_array, float *d_result, void *d_tempStorage,
           size_t tempStorageBytes, int arraySize);
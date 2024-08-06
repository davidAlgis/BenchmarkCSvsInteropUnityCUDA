/**
 * @file action_reduce.cpp
 * @brief Implementation of action_reduce.h
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#include "action_reduce.h"
#include "kernels/reduce.cuh"
#include <chrono>

namespace Benchmark
{

ActionReduce::ActionReduce(void *arrayToGet, int arraySize)
{
    _buffer = CreateBufferInterop(arrayToGet, arraySize);
    _arraySize = arraySize;
    _execTime = 0.0f;
}

int ActionReduce::Start()
{
    h_result = new float(0.0f);
    CUDA_CHECK_RETURN(cudaMalloc(&d_result, sizeof(float)));
    int ret = _buffer->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _arrayResult in CUDA. Abort ActionReduce !");

    ret = _buffer->mapResources<float>(&d_array);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _arrayResult in CUDA. Abort ActionReduce !");
    ret = preAllocationReduce(d_array, d_result, &d_tempStorage,
                              _tempStorageBytes, _arraySize);
    GRUMBLE(ret, "There has been an error during the pre-allocation of reduce. "
                 "Abort ActionReduce !");
    return 0;
}

int ActionReduce::Update()
{
    // We call cudaDeviceSynchronize to make a first synchronization before
    // chrono and to make sure that GPU and CPU are fully synchronize and that
    // the chrono retrieve only the correct time and not other GPU execution
    // time.
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    int ret =
        reduce(d_array, d_result, d_tempStorage, _tempStorageBytes, _arraySize);
    GRUMBLE(ret, "There has been an error during the reduce. "
                 "Abort ActionReduce !");

    CUDA_CHECK_RETURN(
        cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    _execTime = elapsed.count();
    return 0;
}

int ActionReduce::OnDestroy()
{
    int ret = _buffer->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _arrayResult in CUDA. Abort ActionReduce !");
    ret = _buffer->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _arrayResult in CUDA. Abort ActionReduce !");
    delete (h_result);
    // Free temporary storage
    if (d_tempStorage != nullptr)
    {
        CUDA_CHECK_RETURN(cudaFree(d_tempStorage));
    }
    if (d_result != nullptr)
    {
        CUDA_CHECK_RETURN(cudaFree(d_result));
    }
    return 0;
}

} // namespace Benchmark

extern "C"
{
    UNITY_INTERFACE_EXPORT Benchmark::ActionReduce *UNITY_INTERFACE_API
    createActionReduce(void *arrayToSum, int arraySize)
    {
        return (new Benchmark::ActionReduce(arrayToSum, arraySize));
    }

    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCudaReduce(Benchmark::ActionReduce *actionPtr)
    {
        return actionPtr->getExecTime();
    }
}
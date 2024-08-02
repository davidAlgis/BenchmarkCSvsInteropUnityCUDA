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
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    int ret = _buffer->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");

    ret = _buffer->mapResources<float>(&d_array);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    ret = preAllocationReduce(d_array, d_result, d_tempStorage,
                              _tempStorageBytes, _arraySize);
    GRUMBLE(ret, "There has been an error during the pre-allocation of reduce. "
                 "Abort ActionSampleStructBuffer !");
    return 0;
}

int ActionReduce::Update()
{
    auto start = std::chrono::high_resolution_clock::now();

    int ret =
        reduce(d_array, d_result, d_tempStorage, _tempStorageBytes, _arraySize);
    GRUMBLE(ret, "There has been an error during the reduce. "
                 "Abort ActionSampleStructBuffer !");

    CUDA_CHECK_RETURN(
        cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    _execTime = elapsed.count();
    Log::log().debugLog(("result = " + std::to_string(*h_result)).c_str());
    return 0;
}

int ActionReduce::OnDestroy()
{
    int ret = _buffer->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    ret = _buffer->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
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
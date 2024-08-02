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
#include <chrono>
#include <cub/cub.cuh>

namespace Benchmark
{

ActionReduce::ActionReduce(void *arrayToGet, int arraySize)
{
    _buffer = CreateBufferInterop(arrayToGet, arraySize);
    h_result = new float(0.0f);
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    _arraySize = arraySize;
    _execTime = 0.0f;
}

int ActionReduce::Start()
{
    int ret = _buffer->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");

    ret = _buffer->mapResources<float>(&d_array);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");

    // pre allocation for cub::DeviceReduce::Sum see dedicated documentation
    // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html
    void *d_temp_storage = nullptr;
    _temp_storage_bytes = 0;
    CUDA_CHECK_RETURN(cub::DeviceReduce::Sum(
        d_temp_storage, _temp_storage_bytes, d_array, d_result, _arraySize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_temp_storage, _temp_storage_bytes));
    return 0;
}

int ActionReduce::Update()
{
    auto start = std::chrono::high_resolution_clock::now();

    // Sum the elements of the array
    CUDA_CHECK_RETURN(cub::DeviceReduce::Sum(
        d_temp_storage, _temp_storage_bytes, d_array, d_result, _arraySize));

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
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    ret = _buffer->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    delete (h_result);
    // Free temporary storage
    CUDA_CHECK_RETURN(cudaFree(d_temp_storage));
    CUDA_CHECK_RETURN(cudaFree(d_result));
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
    retrieveLastExecTimeCudaGetData(Benchmark::ActionReduce *actionPtr)
    {
        return actionPtr->getExecTime();
    }
}
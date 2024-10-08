/**
 * @file action_get_data.cpp
 * @brief Implementation of action_get_data.h
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#include "action_get_data.h"
#include <chrono>

namespace Benchmark
{

ActionGetData::ActionGetData(void *arrayToGet, int arraySize)
{
    _buffer = CreateBufferInterop(arrayToGet, arraySize);
    h_array = new float[arraySize];
    _arraySize = arraySize;
    _execTime = 0.0f;
}

int ActionGetData::Start()
{
    int ret = _buffer->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _arrayResult in CUDA. Abort ActionGetData !");

    ret = _buffer->mapResources<float>(&d_array);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _arrayResult in CUDA. Abort ActionGetData !");
    return 0;
}

int ActionGetData::Update()
{
    // We call a copy to make a first synchronization before
    // chrono and to make sure that GPU and CPU are fully synchronize and that
    // the chrono retrieve only the correct time and not other GPU execution
    // time. We doesn't use cudaDeviceSynchronize(), as we want to reproduce the
    // same behavior that the unity part, which doesn't have any dedicated
    // synchronize function, and therefore make a single float copy. As the next
    // step will be also a copy it might be faster to launch copy and then copy
    // then doing cudaDeviceSynchronize and then copy.
    CUDA_CHECK_RETURN(
        cudaMemcpy(h_array, d_array, sizeof(float), cudaMemcpyDeviceToHost));
    auto start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK_RETURN(cudaMemcpy(h_array, d_array, sizeof(float) * _arraySize,
                                 cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    _execTime = elapsed.count();
    return 0;
}

int ActionGetData::OnDestroy()
{
    int ret = _buffer->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _arrayResult in CUDA. Abort ActionGetData !");
    ret = _buffer->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _arrayResult in CUDA. Abort ActionGetData !");
    delete[] (h_array);
    return 0;
}

} // namespace Benchmark

extern "C"
{
    UNITY_INTERFACE_EXPORT Benchmark::ActionGetData *UNITY_INTERFACE_API
    createActionGetData(void *arrayToGet, int arraySize)
    {
        return (new Benchmark::ActionGetData(arrayToGet, arraySize));
    }

    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCudaGetData(Benchmark::ActionGetData *actionPtr)
    {
        return actionPtr->getExecTime();
    }
}
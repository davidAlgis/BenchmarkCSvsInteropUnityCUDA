/**
 * @file action_vector_add.cpp
 * @brief Implementation of action_vector_add.h
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#include "action_vector_add.h"
#include "kernels/vector_add.cuh"
#include <chrono>

namespace Benchmark
{

ActionVectorAdd::ActionVectorAdd(void *array1, void *array2, void *arrayResult,
                                 int arraySize)
{
    _buffer1 = CreateBufferInterop(array1, arraySize);
    _buffer2 = CreateBufferInterop(array2, arraySize);
    _bufferResults = CreateBufferInterop(arrayResult, arraySize);
    h_arrayResults = new float[arraySize];
    _arraySize = arraySize;
    _execTime = 0.0f;
}

int ActionVectorAdd::Start()
{
    int ret = _buffer1->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _buffer2->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _bufferResults->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");

    ret = _buffer1->mapResources<float>(&d_array1);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");

    ret = _buffer2->mapResources<float>(&d_array2);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");

    ret = _bufferResults->mapResources<float>(&d_arrayResults);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    return 0;
}

int ActionVectorAdd::Update()
{
    auto start = std::chrono::high_resolution_clock::now();

    kernelCallerWriteBuffer(d_array1, d_array2, d_arrayResults, _arraySize);

    // we only copy a float as in the compute shader part
    CUDA_CHECK_RETURN(cudaMemcpy(h_arrayResults, d_arrayResults, sizeof(float),
                                 cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    _execTime = elapsed.count();
    return 0;
}

int ActionVectorAdd::OnDestroy()
{
    int ret = _buffer1->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _buffer2->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _bufferResults->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");

    ret = _buffer1->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _buffer2->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _bufferResults->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    delete[] (h_arrayResults);
    return 0;
}

} // namespace Benchmark

extern "C"
{
    UNITY_INTERFACE_EXPORT Benchmark::ActionVectorAdd *UNITY_INTERFACE_API
    createActionVectorAdd(void *array1, void *array2, void *arrayResult,
                          int arraySize)
    {
        return (new Benchmark::ActionVectorAdd(array1, array2, arrayResult,
                                               arraySize));
    }

    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCudaVecAdd(Benchmark::ActionVectorAdd *actionPtr)
    {
        return actionPtr->getExecTime();
    }
}
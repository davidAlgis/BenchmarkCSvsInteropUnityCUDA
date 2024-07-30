#include "action_vector_add.h"
#include "kernels/vector_add.cuh"
#include <chrono>

namespace Benchmark
{

ActionVectorAdd::ActionVectorAdd(void *array1, void *array2, void *arrayResult,
                                 int arraySize)
{
    _array1 = CreateBufferInterop(array1, arraySize);
    _array2 = CreateBufferInterop(array2, arraySize);
    d_arrayResult = CreateBufferInterop(arrayResult, arraySize);
    h_arrayResult = new float[arraySize];
    _arraySize = arraySize;
    _execTime = 0.0f;
}

int ActionVectorAdd::Start()
{
    int ret = _array1->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _array2->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");
    ret = d_arrayResult->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    return 0;
}

int ActionVectorAdd::Update()
{
    auto start = std::chrono::high_resolution_clock::now();
    float *array1 = nullptr;
    float *array2 = nullptr;
    float *arrayResult = nullptr;
    int ret = _array1->mapResources<float>(&array1);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");

    ret = _array2->mapResources<float>(&array2);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");

    ret = d_arrayResult->mapResources<float>(&arrayResult);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");

    kernelCallerWriteBuffer(array1, array2, arrayResult, _arraySize);

    ret = _array1->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _array2->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");
    ret = d_arrayResult->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");

    CUDA_CHECK_RETURN(cudaMemcpy(h_arrayResult, arrayResult,
                                 sizeof(float) * _arraySize,
                                 cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    _execTime = elapsed.count();
    return 0;
}

int ActionVectorAdd::OnDestroy()
{
    int ret = _array1->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _array2->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");
    ret = d_arrayResult->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    delete[] (h_arrayResult);
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
    retrieveLastExecTimeCuda(Benchmark::ActionVectorAdd *actionPtr)
    {
        return actionPtr->getExecTime();
    }
}
#include "action_vector_add.h"
#include "kernels/vector_add.cuh"

namespace Benchmark
{

ActionVectorAdd::ActionVectorAdd(void *array1, void *array2, void *arrayResult,
                                 int arraySize)
{
    _array1 = CreateBufferInterop(array1, arraySize);
    _array2 = CreateBufferInterop(array2, arraySize);
    _arrayResult = CreateBufferInterop(arrayResult, arraySize);
    _arraySize = arraySize;
}

int ActionVectorAdd::Start()
{
    int ret = _array1->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _array2->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _arrayResult->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
    return 0;
}

int ActionVectorAdd::Update()
{
    float *array1 = nullptr;
    float *array2 = nullptr;
    float *arrayResult = nullptr;
    int ret = _array1->mapResources<float>(&array1);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");

    ret = _array2->mapResources<float>(&array2);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");

    ret = _arrayResult->mapResources<float>(&arrayResult);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");

    kernelCallerWriteBuffer(array1, array2, arrayResult, _arraySize);

    ret = _array1->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _array1 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _array2->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _array2 in CUDA. Abort ActionSampleStructBuffer !");
    ret = _arrayResult->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
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
    ret = _arrayResult->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _arrayResult in CUDA. Abort ActionSampleStructBuffer !");
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
}
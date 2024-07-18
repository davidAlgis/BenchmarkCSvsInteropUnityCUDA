#include "action_vector_add.h"

namespace Benchmark
{

ActionVectorAdd::ActionVectorAdd(void *bufferPtr, int sizeBuffer)
{
    _structBuffer = CreateBufferInterop(bufferPtr, sizeBuffer);
}

int ActionVectorAdd::Start()
{
    int ret = _structBuffer->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the struct buffer in CUDA. Abort ActionSampleStructBuffer !");
    return 0;
}

int ActionVectorAdd::Update()
{
    float *ptr = nullptr;
    int ret = _structBuffer->mapResources<float>(&ptr);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the struct buffer in CUDA. Abort ActionSampleStructBuffer !");
    // kernelCallerWriteBufferStruct(
    //     *structBuffer->getDimGrid(), *structBuffer->getDimBlock(), ptr,
    //     _structBuffer->getSize(), GetTimeInterop());
    // cudaDeviceSynchronize();
    ret = _structBuffer->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the struct buffer in CUDA. Abort ActionSampleStructBuffer !");
    return 0;
}

int ActionVectorAdd::OnDestroy()
{
    int ret = _structBuffer->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the struct buffer in CUDA. Abort ActionSampleStructBuffer !");
    return 0;
}

} // namespace Benchmark

extern "C"
{
    UNITY_INTERFACE_EXPORT Benchmark::ActionVectorAdd *UNITY_INTERFACE_API
    createActionVectorAdd(void *bufferPtr, int size)
    {
        return (new Benchmark::ActionVectorAdd(bufferPtr, size));
    }
}
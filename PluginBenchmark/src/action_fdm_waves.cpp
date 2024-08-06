/**
 * @file action_reduce.cpp
 * @brief Implementation of action_reduce.h
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#include "action_fdm_waves.h"
#include "kernels/fdm_waves.cuh"
#include <chrono>

namespace Benchmark
{

ActionFDMWaves::ActionFDMWaves(void *htNewPtr, void *htPtr, void *htOldPtr,
                               int width, int height, int depth, float a,
                               float b)
{
    _htNew = CreateTextureInterop(htNewPtr, width, height, depth);
    _ht = CreateTextureInterop(htPtr, width, height, depth);
    _htOld = CreateTextureInterop(htOldPtr, width, height, depth);
    h_pixel = new float(0.0f);
    CUDA_CHECK(cudaMalloc(&d_pixel, sizeof(float)));
    _a = a;
    _b = b;

    _execTime = 0.0f;
}

int ActionFDMWaves::Start()
{
    int ret = _htNew->registerTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _htNew in CUDA. Abort ActionFDMWaves !");
    ret = _ht->registerTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _ht in CUDA. Abort ActionFDMWaves !");
    ret = _htOld->registerTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the _htOld in CUDA. Abort ActionFDMWaves !");

    ret = _htNew->mapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _htNew to surface object in CUDA. Abort "
                 "ActionFDMWaves !");
    ret = _ht->mapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _ht to surface object in CUDA. Abort "
                 "ActionFDMWaves !");
    ret = _htOld->mapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the map of "
                 "the _htOld to surface object in CUDA. Abort "
                 "ActionFDMWaves !");
    return 0;
}

int ActionFDMWaves::Update()
{
    // We call cudaDeviceSynchronize to make a first synchronization before
    // chrono and to make sure that GPU and CPU are fully synchronize and that
    // the chrono retrieve only the correct time and not other GPU execution
    // time.
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    kernelCallerFDMWaves(_htNew->getSurfaceObjectArray(),
                         _htOld->getSurfaceObjectArray(),
                         _ht->getSurfaceObjectArray(), _ht->getWidth(),
                         _ht->getHeight(), _ht->getDepth(), _a, _b);

    kernelCallerSwitchTexReadPixel(
        _htNew->getSurfaceObjectArray(), _htOld->getSurfaceObjectArray(),
        _ht->getSurfaceObjectArray(), _ht->getWidth(), _ht->getHeight(),
        _ht->getDepth(), d_pixel);
    CUDA_CHECK_RETURN(
        cudaMemcpy(h_pixel, d_pixel, sizeof(float), cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    _execTime = elapsed.count();
    return 0;
}

int ActionFDMWaves::OnDestroy()
{
    int ret = _htNew->unmapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _htNew in CUDA. Abort ActionFDMWaves !");
    ret = _ht->unmapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _ht in CUDA. Abort ActionFDMWaves !");
    ret = _htOld->unmapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the _htOld in CUDA. Abort ActionFDMWaves !");

    ret = _htNew->unregisterTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _htNew texture CUDA. Abort "
                 "ActionFDMWaves !");
    ret = _ht->unregisterTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _ht texture CUDA. Abort "
                 "ActionFDMWaves !");
    ret = _htOld->unregisterTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the _htOld texture CUDA. Abort "
                 "ActionFDMWaves !");
    return 0;
}

} // namespace Benchmark

extern "C"
{
    UNITY_INTERFACE_EXPORT Benchmark::ActionFDMWaves *UNITY_INTERFACE_API
    createActionFDMWaves(void *htNewPtr, void *htPtr, void *htOldPtr, int width,
                         int height, int depth, float a, float b)
    {
        return (new Benchmark::ActionFDMWaves(htNewPtr, htPtr, htOldPtr, width,
                                              height, depth, a, b));
    }

    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCudaFDMWaves(Benchmark::ActionFDMWaves *actionPtr)
    {
        return actionPtr->getExecTime();
    }
}
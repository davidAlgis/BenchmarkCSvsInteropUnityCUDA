/**
 * @file action_fdm_waves.h
 * @brief Contains the action to retrieve a vector
 *
 * @author David Algis
 *
 * Company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#pragma once
#include "action.h"
#include "buffer.h"
#include "unity_plugin.h"

namespace Benchmark
{
/**
 * @class ActionFDMWaves
 * @brief A class that solve the wave equation with finite difference method
 * using CUDA and integrates with Unity.
 */
class ActionFDMWaves : Action
{
    public:
    /**
     * @brief Constructs an ActionFDMWaves object.
     * @param arrayToGet Pointer to the graphics native memory pointer of the
     * computer buffer to retrieve.
     * @param arraySize Size of the arrays.
     */
    explicit ActionFDMWaves(void *htNewPtr, void *htPtr, void *htOldPtr,
                            int width, int height, int depth, float a, float b);

    /**
     * @brief Starts the action by registering the buffers in CUDA.
     * @return int Returns 0 on success, error code otherwise.
     */
    int Start() override;

    /**
     * @brief Updates the action by performing the cuda memcpy using CUDA.
     * @return int Returns 0 on success, error code otherwise.
     */
    int Update() override;

    /**
     * @brief Destroys the action by unregistering the buffers in CUDA and
     * cleaning up resources.
     * @return int Returns 0 on success, error code otherwise.
     */
    int OnDestroy() override;

    /**
     * @brief Retrieves the last execution time of the CUDA operation.
     * @return float The last execution time in milliseconds.
     */
    [[nodiscard]] float getExecTime() const
    {
        return _execTime;
    }

    private:
    /**
     * @brief     Pointer to the buffer to retrieve
     */
    Buffer *_buffer;

    /**
     * @brief     Pointer to cuda device array that will be use by CUDA
     */
    float *d_array;

    /**
     * @brief     Pointer to the host array that will be the copy of the device
     * array on host side
     */
    float *h_array;

    /**
     * @brief     Size of the arrays
     */
    int _arraySize;

    /**
     * @brief     Stores the execution time of the CUDA operation
     */
    float _execTime;
};

} // namespace Benchmark

extern "C"
{
    /**
     * @brief Creates an ActionFDMWaves object.
     * @param arrayToGet Pointer to the graphics native memory pointer of the
     * compute buffer to retrieve
     * @param arraySize Size of the arrays.
     * @return Pointer to the created ActionFDMWaves object.
     */
    UNITY_INTERFACE_EXPORT Benchmark::ActionFDMWaves *UNITY_INTERFACE_API
    createActionFDMWaves(void *arrayToGet, int arraySize);

    /**
     * @brief Retrieves the last execution time of the CUDA operation.
     * @param actionPtr Pointer to the ActionFDMWaves object.
     * @return The last execution time in milliseconds.
     */
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCudaFDMWaves(Benchmark::ActionFDMWaves *actionPtr);
}

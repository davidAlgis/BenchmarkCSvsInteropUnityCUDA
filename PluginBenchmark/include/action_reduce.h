/**
 * @file action_reduce.h
 * @brief Contains the action to apply a reduce on a vector
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
 * @class ActionReduce
 * @brief A class that performs reduce using CUDA library : cub and integrates
 * with Unity.
 */
class ActionReduce : Action
{
    public:
    /**
     * @brief Constructs an ActionReduce object.
     * @param arrayToGet Pointer to the graphics native memory pointer of the
     * computer buffer to reduce.
     * @param arraySize Size of the arrays.
     */
    explicit ActionReduce(void *arrayToGet, int arraySize);

    /**
     * @brief Starts the action by registering the buffers in CUDA.
     * @return int Returns 0 on success, error code otherwise.
     */
    int Start() override;

    /**
     * @brief Updates the action by performing the reduce using CUDA.
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
     * @brief     Pointer to cuda device array of temporary storage that will be
     * use by cub to performs the sum.
     */
    void *d_tempStorage;

    /**
     * @brief     Size of d_temp_storage array
     */
    size_t _tempStorageBytes;

    /**
     * @brief     Pointer to the device result of the reduce.
     */
    float *d_result;

    /**
     * @brief     Pointer to the host result that will be the copy of the device
     * array on host side
     */
    float *h_result;

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
     * @brief Creates an ActionReduce object.
     * @param arrayToSum Pointer to the graphics native memory pointer of the
     * compute buffer to sum
     * @param arraySize Size of the arrays.
     * @return Pointer to the created ActionReduce object.
     */
    UNITY_INTERFACE_EXPORT Benchmark::ActionReduce *UNITY_INTERFACE_API
    createActionReduce(void *arrayToSum, int arraySize);

    /**
     * @brief Retrieves the last execution time of the CUDA operation.
     * @param actionPtr Pointer to the ActionReduce object.
     * @return The last execution time in milliseconds.
     */
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCudaReduce(Benchmark::ActionReduce *actionPtr);
}

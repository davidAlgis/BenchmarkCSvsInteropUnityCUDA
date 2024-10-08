/**
 * @file action_vector_add.h
 * @brief Contains the action to add 2 vectors
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
 * @class ActionVectorAdd
 * @brief A class that performs vector addition using CUDA and integrates with
 * Unity.
 */
class ActionVectorAdd : Action
{
    public:
    /**
     * @brief      Constructs an ActionVectorAdd object.
     *
     * @param      array1                Pointer to the graphics native memory
     *                                   pointer of the first computer buffer.
     * @param      array2                Pointer to the graphics native memory
     *                                   pointer of the second computer buffer.
     * @param      arrayResult           Pointer to the graphics native memory
     *                                   pointer of the result computer buffer.
     * @param      arraySize             Size of the arrays.
     * @param[in]  nbrElementToRetrieve  Defined the number of element that
     * needs to be retrieve by CPU from the result compute buffer.
     */
    explicit ActionVectorAdd(void *array1, void *array2, void *arrayResult,
                             int arraySize, int nbrElementToRetrieve);

    /**
     * @brief Starts the action by registering the buffers in CUDA.
     * @return int Returns 0 on success, error code otherwise.
     */
    int Start() override;

    /**
     * @brief Updates the action by performing the vector addition using CUDA.
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
     * @brief     Pointer to the buffer for the first input array
     */
    Buffer *_buffer1;

    /**
     * @brief     Pointer to the buffer for the second input array
     */
    Buffer *_buffer2;

    /**
     * @brief     Pointer to the buffer for the result array
     */
    Buffer *_bufferResults;

    /**
     * @brief     Pointer to cuda device array for the first input array
     */
    float *d_array1;

    /**
     * @brief     Pointer to cuda device array for the second input array
     */
    float *d_array2;

    /**
     * @brief     Pointer to cuda device array for the result array
     */
    float *d_arrayResults;

    /**
     * @brief     Pointer to the host-side result array
     */
    float *h_arrayResults;

    /**
     * @brief     Size of the arrays
     */
    int _arraySize;

    int _nbrElementToRetrieve;

    /**
     * @brief     Stores the execution time of the CUDA operation
     */
    float _execTime;
};

} // namespace Benchmark

extern "C"
{
    /**
     * @brief      Creates an ActionVectorAdd object.
     * @param      array1       Pointer to the graphics native memory pointer of
     *                          the first computer buffer.
     * @param      array2       Pointer to the graphics native memory pointer of
     *                          the second computer buffer.
     * @param      arrayResult  Pointer to the graphics native memory pointer of
     *                          the result computer buffer.
     * @param      arraySize    Size of the arrays.
     *
     * @param[in]  nbrElementToRetrieve  Defined the number of element that
     * needs to be retrieve by CPU from the result compute buffer.
     * @return     Pointer to the created ActionVectorAdd object.
     */
    UNITY_INTERFACE_EXPORT Benchmark::ActionVectorAdd *UNITY_INTERFACE_API
    createActionVectorAdd(void *array1, void *array2, void *arrayResult,
                          int arraySize, int nbrElementToRetrieve);

    /**
     * @brief Retrieves the last execution time of the CUDA operation.
     * @param actionPtr Pointer to the ActionVectorAdd object.
     * @return The last execution time in milliseconds.
     */
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCudaVecAdd(Benchmark::ActionVectorAdd *actionPtr);
}

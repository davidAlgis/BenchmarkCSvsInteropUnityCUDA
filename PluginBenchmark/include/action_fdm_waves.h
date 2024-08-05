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
     * @brief Updates the action by performing the resolution of the waves
     * equation using CUDA.
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
     * @brief     the variable a = c^2 dt^2/dx^2
     */
    float _a;

    /**
     * @brief     2-4a
     */
    float _b;
    /**
     * @brief     Contains the height field at time t
     */
    Texture *_ht;

    /**
     * @brief     Contains the height field at time t-dt
     */
    Texture *_htOld;

    /**
     * @brief     Contains the height field at time t+dt
     */
    Texture *_htNew;

    /**
     * @brief     Stores the execution time of the CUDA operation
     */
    float _execTime;

    /**
     * @brief     Host array that will contains one pixel of ht to sync CPU-GPU
     */
    float *h_pixel;

    /**
     * @brief     Device array that will contains one pixel of ht to sync
     * CPU-GPU
     */
    float *d_pixel;
};

} // namespace Benchmark

extern "C"
{

    UNITY_INTERFACE_EXPORT Benchmark::ActionFDMWaves *UNITY_INTERFACE_API
    createActionFDMWaves(void *htNewPtr, void *htPtr, void *htOldPtr, int width,
                         int height, int depth, float a, float b);

    /**
     * @brief Retrieves the last execution time of the CUDA operation.
     * @param actionPtr Pointer to the ActionFDMWaves object.
     * @return The last execution time in milliseconds.
     */
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCudaFDMWaves(Benchmark::ActionFDMWaves *actionPtr);
}

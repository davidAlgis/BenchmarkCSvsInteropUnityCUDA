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
class ActionVectorAdd : Action
{
    public:
    explicit ActionVectorAdd(void *array1, void *array2, void *arrayResult,
                             int arraySize);
    int Start() override;
    int Update() override;
    int OnDestroy() override;

    [[nodiscard]] float getExecTime() const
    {
        return _execTime;
    }

    private:
    Buffer *_array1;
    Buffer *_array2;
    Buffer *d_arrayResult;

    float *h_arrayResult;
    int _arraySize;

    float _execTime;
};
} // namespace Benchmark

extern "C"
{
    UNITY_INTERFACE_EXPORT Benchmark::ActionVectorAdd *UNITY_INTERFACE_API
    createActionVectorAdd(void *array1, void *array2, void *arrayResult,
                          int arraySize);

    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API
    retrieveLastExecTimeCuda(Benchmark::ActionVectorAdd *actionPtr);
}
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

    private:
    Buffer *_array1;
    Buffer *_array2;
    Buffer *_arrayResult;
    int _arraySize;
};
} // namespace Benchmark

extern "C"
{
    UNITY_INTERFACE_EXPORT Benchmark::ActionVectorAdd *UNITY_INTERFACE_API
    createActionVectorAdd(void *array1, void *array2, void *arrayResult,
                          int arraySize);
}
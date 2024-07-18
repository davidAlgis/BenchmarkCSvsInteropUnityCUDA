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
    explicit ActionVectorAdd(void *bufferPtr, int sizeBuffer);
    int Start() override;
    int Update() override;
    int OnDestroy() override;

    private:
    Buffer *_structBuffer;
};
} // namespace Benchmark

extern "C"
{
    UNITY_INTERFACE_EXPORT Benchmark::ActionVectorAdd *UNITY_INTERFACE_API
    createActionVectorAdd(void *bufferPtr, int size);
}
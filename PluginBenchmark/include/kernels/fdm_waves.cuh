/**
 * @file fdm_waves.cuh
 * @brief Contains the function to call the kernel related to the \ref
 * ActionFDMWaves. Which is solve the waves equation with finite difference
 * method
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#pragma once
#include "texture.h"

void kernelCallerFDMWaves(cudaSurfaceObject_t *htNew,
                          cudaSurfaceObject_t *htOld, cudaSurfaceObject_t *ht,
                          int width, int height, int depth, float a, float b);

void kernelCallerSwitchTexReadPixel(cudaSurfaceObject_t *htNew,
                                    cudaSurfaceObject_t *htOld,
                                    cudaSurfaceObject_t *ht, int width,
                                    int height, int depth, float *d_pixel);
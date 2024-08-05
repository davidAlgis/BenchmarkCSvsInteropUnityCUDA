/**
 * @file vector_add.cu
 * @brief Implementation of the kernel vectorAdd and \ref vector_add.cuh
 *
 * @author David Algis
 *
 * company - Studio Nyx
 * Copyright © Studio Nyx. All rights reserved.
 */
#include "cuda_include.h"
#include "kernels/fdm_waves.cuh"

#define NUM_THREADS_DIM_X 8
#define NUM_THREADS_DIM_Y 8
#define NUM_THREADS_DIM_Z 1

__global__ void wavesFDM(cudaSurfaceObject_t *htNew, cudaSurfaceObject_t *htOld,
                         cudaSurfaceObject_t *ht, int sizeTextureMin1,
                         int depth, float a, float b)
{
    // Calculate the thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Avoid the borders
    if (i <= 0 || i >= sizeTextureMin1 || j <= 0 || j >= sizeTextureMin1 ||
        k >= depth)
    {
        return;
    }

    // Get the neighboring indices
    int iMin1 = i - 1;
    int iPlus1 = i + 1;
    int jMin1 = j - 1;
    int jPlus1 = j + 1;

    float htIPlus1j;
    float htIMin1j;
    float htIjPlus1;
    float htIjMin1;
    float htIj;
    float htOldIj;

    // Read from the surfaces
    surf2Dread(&htIPlus1j, ht[k], iPlus1 * sizeof(float), j);
    surf2Dread(&htIMin1j, ht[k], iMin1 * sizeof(float), j);
    surf2Dread(&htIjPlus1, ht[k], i * sizeof(float), jPlus1);
    surf2Dread(&htIjMin1, ht[k], i * sizeof(float), jMin1);
    surf2Dread(&htIj, ht[k], i * sizeof(float), j);
    surf2Dread(&htOldIj, htOld[k], i * sizeof(float), j);

    // Calculate the new height
    float htNewIj =
        a * (htIPlus1j + htIMin1j + htIjPlus1 + htIjMin1) + b * htIj - htOldIj;

    // Write to the new surface
    surf2Dwrite(htNewIj, htNew[k], i * sizeof(float), j);
}

__global__ void switchTexReadPixel(cudaSurfaceObject_t *htNew,
                                   cudaSurfaceObject_t *htOld,
                                   cudaSurfaceObject_t *ht, int sizeTex,
                                   int depth, float *d_pixel)
{
    // Calculate the thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Avoid the borders
    if (i >= sizeTex || j >= sizeTex || k >= depth)
    {
        return;
    }

    float htNewIJ;
    float htIJ;
    surf2Dread(&htNewIJ, htNew[k], i * sizeof(float), j);
    surf2Dread(&htIJ, ht[k], i * sizeof(float), j);

    surf2Dwrite(htNewIJ, ht[k], i * sizeof(float), j);
    surf2Dwrite(htIJ, htOld[k], i * sizeof(float), j);

    if (i == 0 && j == 0 && k == 0)
    {
        *d_pixel = htNewIJ;
    }
}

void kernelCallerFDMWaves(cudaSurfaceObject_t *htNew,
                          cudaSurfaceObject_t *htOld, cudaSurfaceObject_t *ht,
                          int width, int height, int depth, float a, float b)
{
    // Size of the texture - 1
    int sizeTextureMin1 = width - 1;

    // Define the block and grid dimensions
    dim3 threadsPerBlock(NUM_THREADS_DIM_X, NUM_THREADS_DIM_Y,
                         NUM_THREADS_DIM_Z);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Launch the kernel
    wavesFDM<<<numBlocks, threadsPerBlock>>>(htNew, htOld, ht, sizeTextureMin1,
                                             depth, a, b);
}

void kernelCallerSwitchTexReadPixel(cudaSurfaceObject_t *htNew,
                                    cudaSurfaceObject_t *htOld,
                                    cudaSurfaceObject_t *ht, int width,
                                    int height, int depth, float *d_pixel)
{
    // Define the block and grid dimensions
    dim3 threadsPerBlock(NUM_THREADS_DIM_X, NUM_THREADS_DIM_Y,
                         NUM_THREADS_DIM_Z);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Launch the kernel
    switchTexReadPixel<<<numBlocks, threadsPerBlock>>>(htNew, htOld, ht, width,
                                                       depth, d_pixel);
}
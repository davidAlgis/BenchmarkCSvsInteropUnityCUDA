using System.Diagnostics;
using UnityEngine;

/// <summary>
///     This class performs a resolution of waves equation using Compute shader and measures the execution time.
/// </summary>
public class WavesFDMCS
{
    // Names for compute shader parameters
    private readonly string _aCstName = "A";
    private readonly string _bCstName = "B";
    private readonly string _depthName = "Depth";
    private readonly string _htName = "Ht";
    private readonly string _htNewName = "HtNew";
    private readonly string _htOldName = "HtOld";
    private readonly string _kernelName = "WavesFDM";
    private readonly string _kernelSwitchName = "SwitchTexReadPixel";
    private readonly string _pixelBufferName = "PixelBuffer";
    private readonly string _sizeTextureMin1Name = "SizeTextureMin1";
    private readonly string _sizeTextureName = "Size";
    private ComputeShader _computeShaderFDMWaves;
    private ComputeShader _computeShaderSwitchTexRealPixel;

    // Compute shader and kernel handle
    private int _kernelFDMWavesHandle;
    private int _kernelSwitchHandle;
    private uint _numThreadsX;
    private uint _numThreadsY;

    // Stopwatch for timing GPU execution
    private Stopwatch _stopwatch;

    /// <summary>
    ///     Initializes the compute shaders and sets up necessary parameters.
    /// </summary>
    /// <param name="htNew">The new height texture.</param>
    /// <param name="ht">The current height texture.</param>
    /// <param name="htOld">The old height texture.</param>
    /// <param name="computeShaderFDMWaves">The compute shader for waves FDM.</param>
    /// <param name="computeShaderSwitchPixel">The compute shader for switching textures.</param>
    /// <param name="a">Constant used for waves equation resolution A = c * c * _dt * _dt / (_dx * _dx) </param>
    /// <param name="b">Constant used for waves equation resolution b = 4-2a </param>
    /// <param name="sizeTexture">The size of the texture.</param>
    /// <param name="depth">The depth of the texture array.</param>
    /// <param name="pixelBuffer">The buffer to store the read pixel value.</param>
    public void Init(ref RenderTexture htNew, ref RenderTexture ht, ref RenderTexture htOld,
        ComputeShader computeShaderFDMWaves,
        ComputeShader computeShaderSwitchPixel, float a,
        float b, int sizeTexture, int depth, ComputeBuffer pixelBuffer)
    {
        _computeShaderFDMWaves = computeShaderFDMWaves;
        _computeShaderSwitchTexRealPixel = computeShaderSwitchPixel;
        _kernelFDMWavesHandle = _computeShaderFDMWaves.FindKernel(_kernelName);
        _computeShaderFDMWaves.GetKernelThreadGroupSizes(_kernelFDMWavesHandle, out uint threadGroupSizeX,
            out uint threadGroupSizeY,
            out _);
        _numThreadsX = threadGroupSizeX;
        _numThreadsY = threadGroupSizeY;

        // Set compute shader variables (not height)
        _computeShaderFDMWaves.SetFloat(_aCstName, a);
        _computeShaderFDMWaves.SetFloat(_bCstName, b);
        int sizeTextureMin1 = sizeTexture - 1;
        _computeShaderFDMWaves.SetInt(_sizeTextureMin1Name, sizeTextureMin1);
        _computeShaderFDMWaves.SetInt(_depthName, depth);
        // Set Texture in compute shader
        _computeShaderFDMWaves.SetTexture(_kernelFDMWavesHandle, _htNewName, htNew);
        _computeShaderFDMWaves.SetTexture(_kernelFDMWavesHandle, _htName, ht);
        _computeShaderFDMWaves.SetTexture(_kernelFDMWavesHandle, _htOldName, htOld);

        _kernelSwitchHandle = _computeShaderSwitchTexRealPixel.FindKernel(_kernelSwitchName);
        _computeShaderSwitchTexRealPixel.SetTexture(_kernelSwitchHandle, _htNewName, htNew);
        _computeShaderSwitchTexRealPixel.SetTexture(_kernelSwitchHandle, _htName, ht);
        _computeShaderSwitchTexRealPixel.SetTexture(_kernelSwitchHandle, _htOldName, htOld);
        _computeShaderSwitchTexRealPixel.SetInt(_sizeTextureName, sizeTexture);
        _computeShaderSwitchTexRealPixel.SetInt(_depthName, depth);
        _computeShaderSwitchTexRealPixel.SetBuffer(_kernelSwitchHandle, _pixelBufferName, pixelBuffer);
    }

    /// <summary>
    ///     Updates the compute shader execution and returns the elapsed time.
    /// </summary>
    /// <param name="width">The width of the texture.</param>
    /// <param name="height">The height of the texture.</param>
    /// <param name="depth">The depth of the texture.</param>
    /// <param name="pixelBuffer">The buffer to store the read pixel value.</param>
    /// <param name="result">The array to store the result.</param>
    /// <returns>The elapsed time in milliseconds.</returns>
    public float Update(int width, int height,
        int depth, ComputeBuffer pixelBuffer, ref float[] result)
    {
        // Start the stopwatch
        _stopwatch = Stopwatch.StartNew();

        // Calculate the number of thread groups needed
        int threadGroupsX = Mathf.CeilToInt((float)width / _numThreadsX);
        int threadGroupsY = Mathf.CeilToInt((float)height / _numThreadsY);

        // Dispatch the compute shader
        _computeShaderFDMWaves.Dispatch(_kernelFDMWavesHandle, threadGroupsX, threadGroupsY, depth);

        _computeShaderSwitchTexRealPixel.Dispatch(_kernelSwitchHandle, threadGroupsX, threadGroupsY, depth);
        // retrieve data for synchronize CPU and GPU
        pixelBuffer.GetData(result);
        // Stop the stopwatch and return the elapsed time
        _stopwatch.Stop();
        return (float)_stopwatch.Elapsed.TotalMilliseconds;
    }
}

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
    private readonly string _htName = "Ht";
    private readonly string _htNewName = "HtNew";
    private readonly string _htOldName = "HtOld";
    private readonly string _kernelName = "WavesFDM";
    private readonly string _sizeTextureName = "SizeTextureMin1";

    // Compute shader and kernel handle
    private ComputeShader _computeShader;
    private int _kernelHandle;
    private uint _numThreadsX;
    private uint _numThreadsY;

    // Stopwatch for timing GPU execution
    private Stopwatch _stopwatch;

    private Texture2D _tempTexForSync;

    public void Init(ComputeShader computeShader, float a, float b, int sizeTexture)
    {
        _computeShader = computeShader;
        _kernelHandle = _computeShader.FindKernel(_kernelName);
        _computeShader.GetKernelThreadGroupSizes(_kernelHandle, out uint threadGroupSizeX, out uint threadGroupSizeY,
            out _);
        _numThreadsX = threadGroupSizeX;
        _numThreadsY = threadGroupSizeY;

        // Set compute shader variables (not height)
        _computeShader.SetFloat(_aCstName, a);
        _computeShader.SetFloat(_bCstName, b);
        int sizeTextureMin1 = sizeTexture - 1;
        _computeShader.SetInt(_sizeTextureName, sizeTextureMin1);
        _tempTexForSync = new Texture2D(1, 1, TextureFormat.RFloat, false);
    }

    public float Update(ref RenderTexture htNew, ref RenderTexture ht, ref RenderTexture htOld, int width, int height,
        int depth)
    {
        // Start the stopwatch
        _stopwatch = Stopwatch.StartNew();

        // Calculate the number of thread groups needed
        int threadGroupsX = Mathf.CeilToInt((float)width / _numThreadsX);
        int threadGroupsY = Mathf.CeilToInt((float)height / _numThreadsY);

        // Set Texture in compute shader
        _computeShader.SetTexture(_kernelHandle, _htNewName, htNew);
        _computeShader.SetTexture(_kernelHandle, _htName, ht);
        _computeShader.SetTexture(_kernelHandle, _htOldName, htOld);

        // Dispatch the compute shader
        _computeShader.Dispatch(_kernelHandle, threadGroupsX, threadGroupsY, depth);

        // Retrieve one pixel of the texture to synchronize GPU with CPU
        RenderTexture.active = htNew;
        _tempTexForSync.ReadPixels(new Rect(0, 0, 1, 1), 0, 0);
        _tempTexForSync.Apply();

        // Swap the textures for the next iteration
        (htNew, ht, htOld) = (htOld, htNew, ht);
        // Stop the stopwatch and return the elapsed time
        _stopwatch.Stop();
        return (float)_stopwatch.Elapsed.TotalMilliseconds;
    }
}

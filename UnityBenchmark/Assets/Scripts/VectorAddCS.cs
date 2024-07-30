using System.Diagnostics;
using UnityEngine;

/// <summary>
///     This class performs vector addition using GPU (via compute shader) and measures the execution time.
/// </summary>
public class VectorAddCS
{
    private readonly string _arraySizeName = "ArraySize";
    private readonly string _buffer1Name = "Array1";
    private readonly string _buffer2Name = "Array2";
    private readonly string _bufferResultName = "Result";
    private readonly string _kernelName = "VectorAdd";

    private ComputeShader _computeShader;
    private int _kernelHandle;
    private uint _numThreadsX;

    private Stopwatch _stopwatch;

    public void Init(ComputeShader computeShader, int arraySize, ComputeBuffer buffer1, ComputeBuffer buffer2,
        ComputeBuffer resultBuffer)
    {
        _computeShader = computeShader;
        _kernelHandle = _computeShader.FindKernel(_kernelName);
        _computeShader.GetKernelThreadGroupSizes(_kernelHandle, out uint threadGroupSizeX, out _, out _);
        _numThreadsX = threadGroupSizeX;

        _computeShader.SetBuffer(_kernelHandle, _buffer1Name, buffer1);
        _computeShader.SetBuffer(_kernelHandle, _buffer2Name, buffer2);
        _computeShader.SetBuffer(_kernelHandle, _bufferResultName, resultBuffer);
        _computeShader.SetInt(_arraySizeName, arraySize);
    }

    public float ComputeSum(ComputeBuffer resultBuffer, ref float[] resultArray)
    {
        _stopwatch = Stopwatch.StartNew();

        int threadGroupsX = Mathf.CeilToInt((float)resultArray.Length / _numThreadsX);
        _computeShader.Dispatch(_kernelHandle, threadGroupsX, 1, 1);
        resultBuffer.GetData(resultArray);

        _stopwatch.Stop();
        return (float)_stopwatch.Elapsed.TotalMilliseconds;
    }
}

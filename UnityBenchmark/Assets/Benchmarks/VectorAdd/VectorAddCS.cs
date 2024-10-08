using System.Diagnostics;
using UnityEngine;

/// <summary>
///     This class performs vector addition using Compute shader and measures the execution time.
/// </summary>
public class VectorAddCS
{
    // Names for compute shader parameters
    private readonly string _arraySizeName = "ArraySize";
    private readonly string _buffer1Name = "Array1";
    private readonly string _buffer2Name = "Array2";
    private readonly string _bufferResultName = "Result";
    private readonly string _kernelName = "VectorAdd";

    // Compute shader and kernel handle
    private ComputeShader _computeShader;
    private int _kernelHandle;
    private uint _numThreadsX;

    // Stopwatch for timing GPU execution
    private Stopwatch _stopwatch;

    /// <summary>
    ///     Initializes the compute shader, buffers, and kernel handle.
    /// </summary>
    /// <param name="computeShader">The compute shader to use.</param>
    /// <param name="arraySize">The size of the arrays.</param>
    /// <param name="buffer1">Compute buffer for the first array.</param>
    /// <param name="buffer2">Compute buffer for the second array.</param>
    /// <param name="resultBuffer">Compute buffer for the result array.</param>
    public void Init(ComputeShader computeShader, int arraySize, ComputeBuffer buffer1, ComputeBuffer buffer2,
        ComputeBuffer resultBuffer)
    {
        _computeShader = computeShader;
        _kernelHandle = _computeShader.FindKernel(_kernelName);
        _computeShader.GetKernelThreadGroupSizes(_kernelHandle, out uint threadGroupSizeX, out _, out _);
        _numThreadsX = threadGroupSizeX;

        // Set compute shader buffers and array size
        _computeShader.SetBuffer(_kernelHandle, _buffer1Name, buffer1);
        _computeShader.SetBuffer(_kernelHandle, _buffer2Name, buffer2);
        _computeShader.SetBuffer(_kernelHandle, _bufferResultName, resultBuffer);
        _computeShader.SetInt(_arraySizeName, arraySize);
    }

    /// <summary>
    ///     Computes the sum of two arrays using the compute shader and measures the execution time.
    /// </summary>
    /// <param name="resultBuffer">The compute buffer to store the result.</param>
    /// <param name="resultArray">The array to store the result after computation.</param>
    /// <param name="nbrElementToRetrieve">Indicate the number of element to retrieve on CPU</param>
    /// <returns>The execution time in milliseconds.</returns>
    public float ComputeSum(ComputeBuffer resultBuffer, int arraySize, ref float[] resultArray,
        int nbrElementToRetrieve)
    {
        // Calculate the number of thread groups needed
        int threadGroupsX = Mathf.CeilToInt((float)arraySize / _numThreadsX);
        // we first call warmStep computation to make sur timing is correctly computed
        int warmStep = 5;
        for (int _ = 0; _ < warmStep; _++)
        {
            Computation(threadGroupsX);
        }

        // We call GetData to make a first synchronization before chrono and to make sure that GPU and CPU are fully
        // synchronize and that the chrono retrieve only the correct time and not other GPU execution time.
        resultBuffer.GetData(resultArray, 0, 0, nbrElementToRetrieve);
        // Start the stopwatch
        _stopwatch = Stopwatch.StartNew();

        Computation(threadGroupsX);
        // Retrieve one float from the result buffer, to synchronize the computation
        // (it makes easier profiling, as Unity doesn't have dedicated profiling tools for compute shader)
        resultBuffer.GetData(resultArray, 0, 0, nbrElementToRetrieve);

        // Stop the stopwatch and return the elapsed time
        _stopwatch.Stop();
        return (float)_stopwatch.Elapsed.TotalMilliseconds;
    }

    private void Computation(int threadGroupsX)
    {
        // Dispatch the compute shader
        _computeShader.Dispatch(_kernelHandle, threadGroupsX, 1, 1);
    }
}

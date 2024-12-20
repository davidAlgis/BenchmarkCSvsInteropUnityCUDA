using System.Diagnostics;
using UnityEngine;

/// <summary>
///     This class performs vector addition using Compute shader and measures the execution time.
/// </summary>
public class ReduceCS
{
    // Names for compute shader parameters
    private readonly string _arraySizeName = "sizeArrayToSum";
    private readonly string _arrayToSumName = "arrayToSum";
    private readonly string _kernelName = "Reduce";
    private readonly string _resultReduceName = "resultReduce";
    private readonly string _spinlockName = "spinlock";

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
    /// <param name="arraySize">The size of the array.</param>
    /// <param name="arrayToSum">Compute buffer for the array to sum.</param>
    /// <param name="resultReduce">Compute buffer for the result reduce.</param>
    /// <param name="spinlock">Compute buffer for the spinlock.</param>
    public void Init(ComputeShader computeShader, int arraySize, ComputeBuffer arrayToSum, ComputeBuffer resultReduce,
        ComputeBuffer spinlock)
    {
        _computeShader = computeShader;
        _kernelHandle = _computeShader.FindKernel(_kernelName);
        _computeShader.GetKernelThreadGroupSizes(_kernelHandle, out uint threadGroupSizeX, out _, out _);
        _numThreadsX = threadGroupSizeX;

        // Set compute shader buffers and array size
        _computeShader.SetBuffer(_kernelHandle, _arrayToSumName, arrayToSum);
        _computeShader.SetBuffer(_kernelHandle, _resultReduceName, resultReduce);
        _computeShader.SetBuffer(_kernelHandle, _spinlockName, spinlock);
        _computeShader.SetInt(_arraySizeName, arraySize);
    }

    /// <summary>
    ///     Computes the sum of one arrays using the compute shader and measures the execution time.
    /// </summary>
    /// <param name="resultBuffer">The compute buffer to store the result.</param>
    /// <param name="arraySize">The size of the compute buffer</param>
    /// <param name="resultArray">The array to store the result after computation.</param>
    /// <returns>The execution time in milliseconds.</returns>
    public float ComputeSum(ComputeBuffer resultBuffer, int arraySize, ref float[] resultArray)
    {
        // Calculate the number of thread groups needed
        int threadGroupX = Mathf.CeilToInt((float)arraySize / _numThreadsX);

        // we first call warmStep computation to make sure timing is correctly computed
        int warmStep = 5;
        for (int _ = 0; _ < warmStep; _++)
        {
            Computation(threadGroupX);
        }

        // We call GetData to make a first synchronization before chrono and to make sure that GPU and CPU are fully
        // synchronize and that the chrono retrieve only the correct time and not other GPU execution time.
        resultBuffer.GetData(resultArray, 0, 0, 1);
        // Start the stopwatch
        _stopwatch = Stopwatch.StartNew();

        // Dispatch the compute shader
        Computation(threadGroupX);

        // Retrieve one float from the result buffer, to synchronize the computation
        // (it makes easier profiling, as Unity doesn't have dedicated profiling tools for compute shader)
        resultBuffer.GetData(resultArray, 0, 0, 1);

        // Stop the stopwatch and return the elapsed time
        _stopwatch.Stop();
        return (float)_stopwatch.Elapsed.TotalMilliseconds;
    }

    private void Computation(int threadGroupsX)
    {
        _computeShader.Dispatch(_kernelHandle, threadGroupsX, 1, 1);
    }
}

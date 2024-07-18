using System.Collections.Generic;
using System.Diagnostics;
using TMPro;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
///     This class performs vector addition using GPU (via compute shader) and optionally CPU,
///     then compares the results and profiles the GPU execution time.
/// </summary>
public class VectorAddCS : MonoBehaviour
{
    [SerializeField] private int _arrayLength = 1000;
    [SerializeField] private ComputeShader _computeShader;

    [SerializeField]
    private TextMeshProUGUI _profileText; // Reference to TextMeshPro component for displaying profiling info

    [SerializeField] private bool _compareCPU = true; // Controls whether to compute CPU sum and compare results

    [SerializeField]
    private bool _randomizedEachFrame = true; // Controls whether to compute CPU sum and compare results

    [SerializeField] private float _timeAfterRecord = 5f; // Time in seconds to wait before recording profiling data

    private readonly string _buffer1Name = "Array1";
    private readonly string _buffer2Name = "Array2";
    private readonly string _bufferResultName = "Result";

    // Profiling variables
    private readonly List<float> _executionTimes = new();
    private readonly string _kernelName = "VectorAdd";
    private readonly Stopwatch _stopwatch = new();

    private float[] _array1;
    private float[] _array2;

    private ComputeBuffer _buffer1;
    private ComputeBuffer _buffer2;

    private int _executionCount;
    private int _kernelHandle;

    private uint _numThreadsX;
    private bool _recordingStarted;
    private float[] _resultArray;
    private ComputeBuffer _resultBuffer;

    private float _startTime;
    private float _totalExecutionTime;

    private void Start()
    {
        InitializeBuffers();
        _array1 = GenerateRandomArray(_arrayLength);
        _array2 = GenerateRandomArray(_arrayLength);
        _resultArray = new float[_arrayLength];
        _startTime = Time.time;
        _recordingStarted = false;
    }

    private void Update()
    {
        if (_randomizedEachFrame)
        {
            _array1 = GenerateRandomArray(_arrayLength);
            _array2 = GenerateRandomArray(_arrayLength);
        }

        float gpuExecutionTime = ComputeGPUSum();

        if (_compareCPU)
        {
            float[] cpuSum = ComputeCPUSum();
            bool isEqual = CompareResults(_resultArray, cpuSum);
            if (isEqual == false)
            {
                Debug.LogError("GPU and CPU results do not match!");
            }
        }

        // Check if enough time has passed to start recording profiling information
        if (!_recordingStarted && Time.time - _startTime >= _timeAfterRecord)
        {
            _recordingStarted = true;
        }

        if (_recordingStarted)
        {
            // Update profiling information
            UpdateProfilingInfo(gpuExecutionTime);
        }
    }

    private void OnDisable()
    {
        ReleaseBuffers();
        LogProfilingSummary();
    }

    private void InitializeBuffers()
    {
        _buffer1 = new ComputeBuffer(_arrayLength, sizeof(float));
        _buffer2 = new ComputeBuffer(_arrayLength, sizeof(float));
        _resultBuffer = new ComputeBuffer(_arrayLength, sizeof(float));

        _kernelHandle = _computeShader.FindKernel(_kernelName);
        _computeShader.SetBuffer(_kernelHandle, _buffer1Name, _buffer1);
        _computeShader.SetBuffer(_kernelHandle, _buffer2Name, _buffer2);
        _computeShader.SetBuffer(_kernelHandle, _bufferResultName, _resultBuffer);

        // Get the number of threads per group from the compute shader
        uint threadGroupSizeX, threadGroupSizeY, threadGroupSizeZ;
        _computeShader.GetKernelThreadGroupSizes(_kernelHandle, out threadGroupSizeX, out threadGroupSizeY,
            out threadGroupSizeZ);
        _numThreadsX = threadGroupSizeX;
    }

    private void ReleaseBuffers()
    {
        if (_buffer1 != null)
        {
            _buffer1.Release();
            _buffer2.Release();
            _resultBuffer.Release();
        }
    }

    private float[] GenerateRandomArray(int length)
    {
        float[] array = new float[length];
        for (int i = 0; i < length; i++)
        {
            array[i] = Random.Range(0f, 100f);
        }

        return array;
    }

    /// <summary>
    ///     Computes the GPU sum and measures the execution time.
    /// </summary>
    /// <returns>The execution time in milliseconds.</returns>
    private float ComputeGPUSum()
    {
        _buffer1.SetData(_array1);
        _buffer2.SetData(_array2);

        _stopwatch.Restart();

        // Calculate the number of thread groups needed
        int threadGroupsX = Mathf.CeilToInt((float)_arrayLength / _numThreadsX);

        // Dispatch the compute shader
        _computeShader.Dispatch(_kernelHandle, threadGroupsX, 1, 1);

        // Wait for the GPU to finish
        _resultBuffer.GetData(_resultArray);

        _stopwatch.Stop();
        float elapsedTimeMs = (float)_stopwatch.Elapsed.TotalMilliseconds;

        return elapsedTimeMs;
    }

    private float[] ComputeCPUSum()
    {
        float[] result = new float[_arrayLength];
        for (int i = 0; i < _arrayLength; i++)
        {
            result[i] = _array1[i] + _array2[i];
        }

        return result;
    }

    private bool CompareResults(float[] gpuResult, float[] cpuResult)
    {
        bool isEqual = true;
        for (int i = 0; i < _arrayLength; i++)
        {
            if (Mathf.Abs(gpuResult[i] - cpuResult[i]) > Mathf.Epsilon)
            {
                isEqual = false;
                Debug.LogError($"Mismatch at index {i}: GPU = {gpuResult[i]}, CPU = {cpuResult[i]}");
            }
        }

        return isEqual;
    }

    /// <summary>
    ///     Updates the profiling information and displays it on screen.
    /// </summary>
    /// <param name="executionTime">The latest execution time of the compute shader.</param>
    private void UpdateProfilingInfo(float executionTime)
    {
        _executionTimes.Add(executionTime);
        _totalExecutionTime += executionTime;
        _executionCount++;

        float averageTime = _totalExecutionTime / _executionCount;

        if (_profileText != null)
        {
            _profileText.text = $"Average Time CS : {averageTime:F3} ms";
        }
    }

    /// <summary>
    ///     Logs the profiling summary when the GameObject is destroyed.
    /// </summary>
    private void LogProfilingSummary()
    {
        if (_executionCount > 0)
        {
            float overallAverage = _totalExecutionTime / _executionCount;

            // Compute variance
            float sumOfSquaredDifferences = 0f;
            foreach (float time in _executionTimes)
            {
                float difference = time - overallAverage;
                sumOfSquaredDifferences += difference * difference;
            }

            float variance = sumOfSquaredDifferences / _executionCount;

            float standardDeviation = Mathf.Sqrt(variance);

            Debug.Log("Compute Shader Profiling Summary:" +
                      $"Samples: {_executionCount}" +
                      $", Average Execution Time: {overallAverage:F3} ms" +
                      $", Standard Deviation: {standardDeviation:F3} ms" +
                      $", Variance: {variance:F3} ms^2");
        }
    }
}

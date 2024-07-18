using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using TMPro;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
///     This class performs vector addition using GPU (via compute shader) and optionally CPU,
///     then compares the results and profiles the GPU execution time.
/// </summary>
public class VectorAddCS : MonoBehaviour
{
    [SerializeField] private List<int> _arraySizes = new() { 1000, 10000, 100000 };
    [SerializeField] private int _numSamplesPerSize = 10000;
    [SerializeField] private ComputeShader _computeShader;

    [SerializeField]
    private TextMeshProUGUI _profileText; // Reference to TextMeshPro component for displaying profiling info

    [SerializeField]
    private TextMeshProUGUI _sizeArrayText; // Reference to TextMeshPro component for displaying array size

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

    private readonly List<ProfilingData> _profilingResults = new();
    private readonly Stopwatch _stopwatch = new();

    private float[] _array1;
    private float[] _array2;

    private ComputeBuffer _buffer1;
    private ComputeBuffer _buffer2;

    private int _currentArraySizeIndex;
    private int _currentSampleCount;

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
        _startTime = Time.time;
        _recordingStarted = false;
        InitializeBuffers();
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
    }

    private void Update()
    {
        if (!_recordingStarted && Time.time - _startTime >= _timeAfterRecord)
        {
            _recordingStarted = true;
        }

        if (_recordingStarted && _currentSampleCount < _numSamplesPerSize)
        {
            if (_randomizedEachFrame)
            {
                _array1 = GenerateRandomArray(_arraySizes[_currentArraySizeIndex]);
                _array2 = GenerateRandomArray(_arraySizes[_currentArraySizeIndex]);
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

            // Update profiling information
            UpdateProfilingInfo(gpuExecutionTime);

            _currentSampleCount++;
        }
        else if (_currentSampleCount >= _numSamplesPerSize)
        {
            LogProfilingSummary();
            SaveProfilingData();

            _currentSampleCount = 0;
            _executionTimes.Clear();
            _totalExecutionTime = 0;
            _executionCount = 0;

            _currentArraySizeIndex++;

            if (_currentArraySizeIndex < _arraySizes.Count)
            {
                ReleaseBuffers();
                InitializeArrays(_arraySizes[_currentArraySizeIndex]);
            }
            else
            {
                Debug.Log("All tests completed.");
                ExportToCsv();
                enabled = false; // Stop the script
            }
        }
    }

    private void InitializeBuffers()
    {
        _kernelHandle = _computeShader.FindKernel(_kernelName);

        // Get the number of threads per group from the compute shader
        uint threadGroupSizeX;
        _computeShader.GetKernelThreadGroupSizes(_kernelHandle, out threadGroupSizeX, out _,
            out _);
        _numThreadsX = threadGroupSizeX;
    }

    private void InitializeArrays(int arraySize)
    {
        _array1 = GenerateRandomArray(arraySize);
        _array2 = GenerateRandomArray(arraySize);
        _resultArray = new float[arraySize];

        _buffer1 = new ComputeBuffer(arraySize, sizeof(float));
        _buffer2 = new ComputeBuffer(arraySize, sizeof(float));
        _resultBuffer = new ComputeBuffer(arraySize, sizeof(float));

        _computeShader.SetBuffer(_kernelHandle, _buffer1Name, _buffer1);
        _computeShader.SetBuffer(_kernelHandle, _buffer2Name, _buffer2);
        _computeShader.SetBuffer(_kernelHandle, _bufferResultName, _resultBuffer);

        if (_sizeArrayText != null)
        {
            _sizeArrayText.text = $"Array Size: {arraySize}";
        }
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
        int threadGroupsX = Mathf.CeilToInt((float)_arraySizes[_currentArraySizeIndex] / _numThreadsX);

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
        float[] result = new float[_arraySizes[_currentArraySizeIndex]];
        for (int i = 0; i < _arraySizes[_currentArraySizeIndex]; i++)
        {
            result[i] = _array1[i] + _array2[i];
        }

        return result;
    }

    private bool CompareResults(float[] gpuResult, float[] cpuResult)
    {
        bool isEqual = true;
        for (int i = 0; i < _arraySizes[_currentArraySizeIndex]; i++)
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
            _profileText.text =
                $"Average Time CS: {averageTime:F3} ms";
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

            Debug.Log($"Compute Shader Profiling Summary for Array Size {_arraySizes[_currentArraySizeIndex]}:" +
                      $" Samples: {_executionCount}" +
                      $", Average Execution Time: {overallAverage:F3} ms" +
                      $", Standard Deviation: {standardDeviation:F3} ms" +
                      $", Variance: {variance:F3} ms^2");

            // Save profiling data to list
            _profilingResults.Add(new ProfilingData
            {
                ArraySize = _arraySizes[_currentArraySizeIndex],
                AverageExecutionTime = overallAverage,
                StandardDeviation = standardDeviation,
                Variance = variance,
                SampleCount = _executionCount
            });
        }
    }

    private void SaveProfilingData()
    {
        // Save the profiling data for the current array size
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

        _profilingResults.Add(new ProfilingData
        {
            ArraySize = _arraySizes[_currentArraySizeIndex],
            AverageExecutionTime = overallAverage,
            StandardDeviation = standardDeviation,
            Variance = variance,
            SampleCount = _executionCount
        });
    }

    private void ExportToCsv()
    {
        string filePath = Path.Combine(Application.dataPath, "ProfilingResults.csv");
        using (StreamWriter writer = new(filePath))
        {
            writer.WriteLine("ArraySize;SampleCount;AverageExecutionTime;StandardDeviation;Variance");
            foreach (ProfilingData data in _profilingResults)
            {
                writer.WriteLine(
                    $"{data.ArraySize};{data.SampleCount};{data.AverageExecutionTime:F3};{data.StandardDeviation:F3};{data.Variance:F3}");
            }
        }

        Debug.Log($"Profiling results exported to {filePath}");
    }
}

public class ProfilingData
{
    public int ArraySize { get; set; }

    public int SampleCount { get; set; }

    public float AverageExecutionTime { get; set; }

    public float StandardDeviation { get; set; }

    public float Variance { get; set; }
}

using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;

/// <summary>
///     This class manages the vector addition process and profiling.
/// </summary>
public class VectorAddManager : MonoBehaviour
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
    private readonly List<float> _executionTimes = new();
    private readonly List<ProfilingDataSum> _profilingResults = new();

    private float[] _array1;
    private float[] _array2;

    private ComputeBuffer _buffer1;
    private ComputeBuffer _buffer2;
    private int _currentArraySizeIndex;
    private int _currentSampleCount;
    private int _executionCount;
    private bool _recordingStarted;
    private float[] _resultArray;
    private ComputeBuffer _resultBuffer;
    private float _startTime;
    private float _totalExecutionTime;

    private VectorAddCS _vectorAddCompute;

    private void Start()
    {
        _startTime = Time.time;
        _recordingStarted = false;
        _vectorAddCompute = new VectorAddCS();
        InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
        _vectorAddCompute.Init(_computeShader, _arraySizes[_currentArraySizeIndex], _buffer1, _buffer2, _resultBuffer);
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
                InitializeArrays(_arraySizes[_currentArraySizeIndex]);
            }

            float gpuExecutionTime = _vectorAddCompute.ComputeSum(_resultBuffer, ref _resultArray);

            if (_compareCPU)
            {
                float[] cpuSum = ComputeCPUSum();
                bool isEqual = CompareResults(cpuSum);
                if (isEqual == false)
                {
                    Debug.LogError("GPU and CPU results do not match!");
                }
            }

            UpdateProfilingInfo(gpuExecutionTime);
            _currentSampleCount++;
        }
        else if (_currentSampleCount >= _numSamplesPerSize)
        {
            LogProfilingSummary();
            SaveProfilingDataSum();

            _currentSampleCount = 0;
            _executionTimes.Clear();
            _totalExecutionTime = 0;
            _executionCount = 0;

            _currentArraySizeIndex++;

            if (_currentArraySizeIndex < _arraySizes.Count)
            {
                ReleaseBuffers();
                InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
                InitializeArrays(_arraySizes[_currentArraySizeIndex]);
                _vectorAddCompute.Init(_computeShader, _arraySizes[_currentArraySizeIndex], _buffer1, _buffer2,
                    _resultBuffer);
            }
            else
            {
                Debug.Log("All tests completed.");
                ExportToCsv();
                enabled = false; // Stop the script
            }
        }
    }

    private void InitializeArrays(int arraySize)
    {
        _array1 = GenerateRandomArray(arraySize);
        _array2 = GenerateRandomArray(arraySize);
        _buffer1.SetData(_array1);
        _buffer2.SetData(_array2);
        _resultArray = new float[arraySize];
    }

    private void InitializeBuffers(int arraySize)
    {
        _buffer1 = new ComputeBuffer(arraySize, sizeof(float));
        _buffer2 = new ComputeBuffer(arraySize, sizeof(float));
        _resultBuffer = new ComputeBuffer(arraySize, sizeof(float));
    }

    private void ReleaseBuffers()
    {
        _buffer1.Release();
        _buffer2.Release();
        _resultBuffer.Release();
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

    private float[] ComputeCPUSum()
    {
        float[] result = new float[_array1.Length];
        for (int i = 0; i < _array1.Length; i++)
        {
            result[i] = _array1[i] + _array2[i];
        }

        return result;
    }

    private bool CompareResults(float[] cpuResult)
    {
        bool isEqual = true;
        for (int i = 0; i < _resultArray.Length; i++)
        {
            if (Mathf.Abs(_resultArray[i] - cpuResult[i]) > Mathf.Epsilon)
            {
                isEqual = false;
                Debug.LogError($"Mismatch at index {i}: GPU = {_resultArray[i]}, CPU = {cpuResult[i]}");
            }
        }

        return isEqual;
    }

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

    private void LogProfilingSummary()
    {
        if (_executionCount > 0)
        {
            float overallAverage = _totalExecutionTime / _executionCount;

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

            _profilingResults.Add(new ProfilingDataSum
            {
                ArraySize = _arraySizes[_currentArraySizeIndex],
                AverageExecutionTime = overallAverage,
                StandardDeviation = standardDeviation,
                Variance = variance,
                SampleCount = _executionCount
            });
        }
    }

    private void SaveProfilingDataSum()
    {
        float overallAverage = _totalExecutionTime / _executionCount;

        float sumOfSquaredDifferences = 0f;
        foreach (float time in _executionTimes)
        {
            float difference = time - overallAverage;
            sumOfSquaredDifferences += difference * difference;
        }

        float variance = sumOfSquaredDifferences / _executionCount;
        float standardDeviation = Mathf.Sqrt(variance);

        _profilingResults.Add(new ProfilingDataSum
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
            foreach (ProfilingDataSum data in _profilingResults)
            {
                writer.WriteLine(
                    $"{data.ArraySize};{data.SampleCount};{data.AverageExecutionTime:F3};{data.StandardDeviation:F3};{data.Variance:F3}");
            }
        }

        Debug.Log($"Profiling results exported to {filePath}");
    }
}

public class ProfilingDataSum
{
    public int ArraySize { get; set; }

    public int SampleCount { get; set; }

    public float AverageExecutionTime { get; set; }

    public float StandardDeviation { get; set; }

    public float Variance { get; set; }
}

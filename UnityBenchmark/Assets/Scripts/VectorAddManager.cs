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
    [SerializeField] private VectorAddCUDA _vectorAddCuda;

    [SerializeField]
    private TextMeshProUGUI _profileTextCS; // Reference to TextMeshPro component for displaying CS profiling info

    [SerializeField]
    private TextMeshProUGUI _profileTextCUDA; // Reference to TextMeshPro component for displaying CUDA profiling info

    [SerializeField]
    private TextMeshProUGUI _profileTextDifference; // Reference to TextMeshPro component for displaying difference info

    [SerializeField]
    private TextMeshProUGUI _sizeArrayText; // Reference to TextMeshPro component for displaying array size

    [SerializeField] private bool _compareCPU = true; // Controls whether to compute CPU sum and compare results

    [SerializeField] private bool _randomizedEachFrame = true; // Controls whether to randomize arrays each frame

    [SerializeField] private float _timeAfterRecord = 5f; // Time in seconds to wait before recording profiling data
    private readonly List<float> _executionTimesCS = new();
    private readonly List<float> _executionTimesCUDA = new();
    private readonly List<ProfilingDataSum> _profilingResults = new();

    private float[] _array1;
    private float[] _array2;

    private ComputeBuffer _buffer1;
    private ComputeBuffer _buffer2;
    private int _currentArraySizeIndex;
    private int _currentSampleCount;
    private bool _endTest;
    private int _executionCount;
    private bool _recordingStarted;
    private float[] _resultArray;
    private ComputeBuffer _resultBuffer;
    private float _startTime;
    private float _totalExecutionTimeCS;
    private float _totalExecutionTimeCUDA;

    private VectorAddCS _vectorAddCompute;

    private void Start()
    {
        _vectorAddCuda.InitializeInteropHandler();
        _startTime = Time.time;
        _recordingStarted = false;
        _vectorAddCompute = new VectorAddCS();
        InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
        _vectorAddCompute.Init(_computeShader, _arraySizes[_currentArraySizeIndex], _buffer1, _buffer2, _resultBuffer);
        _vectorAddCuda.InitializeActionsAdd(_arraySizes[_currentArraySizeIndex], _buffer1, _buffer2, _resultBuffer);
        // we execute vector add cuda once to have one execution time at least as the time retrieve is the one from the last frame
        _vectorAddCuda.ComputeSum();
    }

    private void Update()
    {
        if (_endTest)
        {
            _sizeArrayText.text = "Vector Add is done";
            return;
        }

        if (!_recordingStarted && Time.time - _startTime >= _timeAfterRecord)
        {
            _recordingStarted = true;
        }

        if (_recordingStarted == false)
        {
            _buffer1.SetData(_array1);
            _buffer2.SetData(_array2);

            _vectorAddCompute.ComputeSum(_resultBuffer, ref _resultArray);
            _vectorAddCuda.ComputeSum();
            _profileTextCS.text = "Average Time CS: loading...";
            _profileTextCUDA.text = "Average Time CUDA: loading...";
            _profileTextDifference.text = "Difference: loading...";
        }

        if (_recordingStarted && _currentSampleCount < _numSamplesPerSize)
        {
            int arraySize = _arraySizes[_currentArraySizeIndex];
            _sizeArrayText.text = "Vector Add - " + arraySize + " - Sample " + _currentSampleCount + "/" +
                                  _numSamplesPerSize;
            if (_randomizedEachFrame)
            {
                InitializeArrays(arraySize);
            }

            _buffer1.SetData(_array1);
            _buffer2.SetData(_array2);
            float gpuExecutionTimeCS = _vectorAddCompute.ComputeSum(_resultBuffer, ref _resultArray);
            float gpuExecutionTimeCUDA = _vectorAddCuda.ComputeSum();

            if (_compareCPU)
            {
                float[] cpuSum = ComputeCPUSum();
                bool isEqual = CompareResults(cpuSum);
                if (!isEqual)
                {
                    Debug.LogError("GPU and CPU results do not match!");
                }
            }

            UpdateProfilingInfo(gpuExecutionTimeCS, gpuExecutionTimeCUDA);
            _currentSampleCount++;
        }
        else if (_currentSampleCount >= _numSamplesPerSize)
        {
            LogProfilingSummary();
            // SaveProfilingDataSum();

            _currentSampleCount = 0;
            _executionTimesCS.Clear();
            _executionTimesCUDA.Clear();
            _totalExecutionTimeCS = 0;
            _totalExecutionTimeCUDA = 0;
            _executionCount = 0;

            ReleaseBuffers();
            _currentArraySizeIndex++;

            if (_currentArraySizeIndex < _arraySizes.Count)
            {
                InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
                InitializeArrays(_arraySizes[_currentArraySizeIndex]);
                _vectorAddCompute.Init(_computeShader, _arraySizes[_currentArraySizeIndex], _buffer1, _buffer2,
                    _resultBuffer);
                _vectorAddCuda.InitializeActionsAdd(_arraySizes[_currentArraySizeIndex], _buffer1, _buffer2,
                    _resultBuffer);
            }
            else
            {
                Debug.Log("All tests completed.");
                ExportToCsv();
                Application.Quit();
                _endTest = true;
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
        _vectorAddCuda.DestroyActionsAdd();
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

    private void UpdateProfilingInfo(float executionTimeCS, float executionTimeCUDA)
    {
        _executionTimesCS.Add(executionTimeCS);
        _executionTimesCUDA.Add(executionTimeCUDA);
        _totalExecutionTimeCS += executionTimeCS;
        _totalExecutionTimeCUDA += executionTimeCUDA;
        _executionCount++;

        float averageTimeCS = _totalExecutionTimeCS / _executionCount;
        float averageTimeCUDA = _totalExecutionTimeCUDA / _executionCount;
        float difference = averageTimeCS - averageTimeCUDA;

        if (_profileTextCS != null)
        {
            _profileTextCS.text = $"Average Time CS: {averageTimeCS:F3} ms";
        }

        if (_profileTextCUDA != null)
        {
            _profileTextCUDA.text = $"Average Time CUDA: {averageTimeCUDA:F3} ms";
        }

        if (_profileTextDifference != null)
        {
            _profileTextDifference.text = $"Difference: {difference:F3} ms";
        }
    }

    private void LogProfilingSummary()
    {
        if (_executionCount > 0)
        {
            float overallAverageCS = _totalExecutionTimeCS / _executionCount;
            float overallAverageCUDA = _totalExecutionTimeCUDA / _executionCount;

            float sumOfSquaredDifferencesCS = 0f;
            foreach (float time in _executionTimesCS)
            {
                float difference = time - overallAverageCS;
                sumOfSquaredDifferencesCS += difference * difference;
            }

            float varianceCS = sumOfSquaredDifferencesCS / _executionCount;
            float standardDeviationCS = Mathf.Sqrt(varianceCS);

            float sumOfSquaredDifferencesCUDA = 0f;
            foreach (float time in _executionTimesCUDA)
            {
                float difference = time - overallAverageCUDA;
                sumOfSquaredDifferencesCUDA += difference * difference;
            }

            float varianceCUDA = sumOfSquaredDifferencesCUDA / _executionCount;
            float standardDeviationCUDA = Mathf.Sqrt(varianceCUDA);

            Debug.Log($"Compute Shader Profiling Summary for Array Size {_arraySizes[_currentArraySizeIndex]}:" +
                      $" Samples: {_executionCount}" +
                      $", Average Execution Time (CS): {overallAverageCS:F3} ms" +
                      $", Standard Deviation (CS): {standardDeviationCS:F3} ms" +
                      $", Variance (CS): {varianceCS:F3} ms^2");

            Debug.Log($"CUDA Profiling Summary for Array Size {_arraySizes[_currentArraySizeIndex]}:" +
                      $" Samples: {_executionCount}" +
                      $", Average Execution Time (CUDA): {overallAverageCUDA:F3} ms" +
                      $", Standard Deviation (CUDA): {standardDeviationCUDA:F3} ms" +
                      $", Variance (CUDA): {varianceCUDA:F3} ms^2");

            _profilingResults.Add(new ProfilingDataSum
            {
                ArraySize = _arraySizes[_currentArraySizeIndex],
                AverageExecutionTimeCS = overallAverageCS,
                StandardDeviationCS = standardDeviationCS,
                VarianceCS = varianceCS,
                AverageExecutionTimeCUDA = overallAverageCUDA,
                StandardDeviationCUDA = standardDeviationCUDA,
                VarianceCUDA = varianceCUDA,
                SampleCount = _executionCount
            });
        }
    }

    private void SaveProfilingDataSum()
    {
        float overallAverageCS = _totalExecutionTimeCS / _executionCount;
        float overallAverageCUDA = _totalExecutionTimeCUDA / _executionCount;

        float sumOfSquaredDifferencesCS = 0f;
        foreach (float time in _executionTimesCS)
        {
            float difference = time - overallAverageCS;
            sumOfSquaredDifferencesCS += difference * difference;
        }

        float varianceCS = sumOfSquaredDifferencesCS / _executionCount;
        float standardDeviationCS = Mathf.Sqrt(varianceCS);

        float sumOfSquaredDifferencesCUDA = 0f;
        foreach (float time in _executionTimesCUDA)
        {
            float difference = time - overallAverageCUDA;
            sumOfSquaredDifferencesCUDA += difference * difference;
        }

        float varianceCUDA = sumOfSquaredDifferencesCUDA / _executionCount;
        float standardDeviationCUDA = Mathf.Sqrt(varianceCUDA);

        _profilingResults.Add(new ProfilingDataSum
        {
            ArraySize = _arraySizes[_currentArraySizeIndex],
            AverageExecutionTimeCS = overallAverageCS,
            StandardDeviationCS = standardDeviationCS,
            VarianceCS = varianceCS,
            AverageExecutionTimeCUDA = overallAverageCUDA,
            StandardDeviationCUDA = standardDeviationCUDA,
            VarianceCUDA = varianceCUDA,
            SampleCount = _executionCount
        });
    }

    private void ExportToCsv()
    {
        string filePath = Path.Combine(Application.dataPath, "ProfilingResults.csv");
        using (StreamWriter writer = new(filePath))
        {
            writer.WriteLine(
                "ArraySize;SampleCount;AverageExecutionTimeCS;StandardDeviationCS;VarianceCS;AverageExecutionTimeCUDA;StandardDeviationCUDA;VarianceCUDA;Difference");
            foreach (ProfilingDataSum data in _profilingResults)
            {
                float difference = data.AverageExecutionTimeCS - data.AverageExecutionTimeCUDA;
                writer.WriteLine(
                    $"{data.ArraySize};{data.SampleCount};{data.AverageExecutionTimeCS:F3};{data.StandardDeviationCS:F3};{data.VarianceCS:F3};{data.AverageExecutionTimeCUDA:F3};{data.StandardDeviationCUDA:F3};{data.VarianceCUDA:F3};{difference:F3}");
            }
        }

        Debug.Log($"Profiling results exported to {filePath}");
    }
}

public class ProfilingDataSum
{
    public int ArraySize { get; set; }

    public int SampleCount { get; set; }

    public float AverageExecutionTimeCS { get; set; }

    public float StandardDeviationCS { get; set; }

    public float VarianceCS { get; set; }

    public float AverageExecutionTimeCUDA { get; set; }

    public float StandardDeviationCUDA { get; set; }

    public float VarianceCUDA { get; set; }
}

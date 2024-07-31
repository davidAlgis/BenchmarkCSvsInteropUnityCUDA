using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.Serialization;

/// <summary>
///     This class manages the vector addition process and profiling.
/// </summary>
public class VectorAddManager : MonoBehaviour
{
    // Configuration parameters for array sizes and samples per size
    [SerializeField] private List<int> _arraySizes = new()
    {
        1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000, 2500000, 5000000,
        7500000, 10000000
    };

    [SerializeField] private int _numSamplesPerSize = 10000;

    // References to compute shader and CUDA class
    [SerializeField] private ComputeShader _computeShader;
    [SerializeField] private VectorAddCUDA _vectorAddCuda;

    // UI components for displaying profiling information
    // Reference to TextMeshPro component for displaying CS profiling info
    [SerializeField] private TextMeshProUGUI _profileTextCS;

    // Reference to TextMeshPro component for displaying CUDA profiling info
    [SerializeField] private TextMeshProUGUI _profileTextCUDA;

    // Reference to TextMeshPro component for displaying difference info
    [SerializeField] private TextMeshProUGUI _profileTextDifference;

    // Reference to TextMeshPro component for displaying array size
    [SerializeField] private TextMeshProUGUI _sizeArrayText;

    // Flags for controlling CPU comparison and array randomization
    // Controls whether to compute CPU sum and compare results
    [SerializeField] private bool _compareCPU;

    // Controls whether to randomize arrays each frame
    [SerializeField] private bool _randomizedEachFrame;

    // Time to wait before recording profiling data
    [FormerlySerializedAs("_timeAfterRecord")] [SerializeField]
    private float _waitTimeBeforeClocking = 5f;

    // Lists to store execution times and profiling results
    private readonly List<float> _executionTimesCS = new();
    private readonly List<float> _executionTimesCUDA = new();
    private readonly List<ProfilingDataSum> _profilingResults = new();

    // Arrays for computation
    private float[] _array1;
    private float[] _array2;

    // Compute buffers for GPU computation
    private ComputeBuffer _buffer1;
    private ComputeBuffer _buffer2;

    // Indices and counters for tracking progress
    private int _currentArraySizeIndex;
    private int _currentSampleCount;
    private int _executionCount;
    private bool _recordingStarted;

    // Arrays for storing results and timings
    private float[] _resultArray;
    private ComputeBuffer _resultBuffer;
    private float _startTime;
    private float _totalExecutionTimeCS;
    private float _totalExecutionTimeCUDA;

    // Compute shader class for vector addition
    private VectorAddCS _vectorAddCompute;

    /// <summary>
    ///     Initializes the components and starts the profiling process.
    /// </summary>
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

        // Execute CUDA vector addition once to get an initial execution time
        // (as we retrieve the variable with one frame late, if we don't do that we will retrieve 0 for the first frame)
        _vectorAddCuda.ComputeSum();
    }

    /// <summary>
    ///     Updates the profiling process at each frame.
    /// </summary>
    private void Update()
    {
        // the timer recorded with gpu at the beginning of execution are most of the time not representative.
        // Therefore, we wait a few seconds before clocking the time of the computations.
        if (!_recordingStarted)
        {
            _buffer1.SetData(_array1);
            _buffer2.SetData(_array2);

            _vectorAddCompute.ComputeSum(_resultBuffer, ref _resultArray);
            _vectorAddCuda.ComputeSum();
            _profileTextCS.text = "Average Time CS: loading...";
            _profileTextCUDA.text = "Average Time CUDA: loading...";
            _profileTextDifference.text = "Difference: loading...";

            // launch record when we reach the _timeAfterRecord
            if (Time.time - _startTime >= _waitTimeBeforeClocking)
            {
                _recordingStarted = true;
            }
        }

        if (_recordingStarted && _currentSampleCount < _numSamplesPerSize)
        {
            int arraySize = _arraySizes[_currentArraySizeIndex];
            _sizeArrayText.text = $"Vector Add - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";

            if (_randomizedEachFrame)
            {
                InitializeArrays(arraySize);
            }

            // The core of the computation 
            // Execute sum and retrieve data with compute shader
            float gpuExecutionTimeCS = _vectorAddCompute.ComputeSum(_resultBuffer, ref _resultArray);
            // Execute sum and retrieve data with CUDA
            float gpuExecutionTimeCUDA = _vectorAddCuda.ComputeSum();

            // To check results
            if (_compareCPU)
            {
                float[] cpuSum = ComputeCPUSum();
                if (!CompareResults(cpuSum))
                {
                    Debug.LogError("GPU and CPU results do not match!");
                }
            }

            // Update the UI 
            UpdateProfilingInfo(gpuExecutionTimeCS, gpuExecutionTimeCUDA);
            _currentSampleCount++;
        }
        // When we reach the number of sample required
        else if (_currentSampleCount >= _numSamplesPerSize)
        {
            // log the data associated to the test
            LogProfilingSummary();

            _currentSampleCount = 0;
            _executionTimesCS.Clear();
            _executionTimesCUDA.Clear();
            _totalExecutionTimeCS = 0;
            _totalExecutionTimeCUDA = 0;
            _executionCount = 0;

            // release memory
            ReleaseBuffers();
            _currentArraySizeIndex++;

            // if we are not at the end of the array of size, we re-initialize the array with the new size of array
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
                // if we are at the end we export all the data to a csv and stop the application
                Debug.Log("All tests completed. Export to csv the result.");
                ExportToCsv();
                // we stop the application
#if UNITY_EDITOR || DEVELOPMENT_BUILD
                EditorApplication.isPlaying = false;
#else
                Application.Quit();
#endif
            }
        }
    }

    /// <summary>
    ///     Initializes the arrays for computation.
    /// </summary>
    /// <param name="arraySize">The size of the arrays to initialize.</param>
    private void InitializeArrays(int arraySize)
    {
        // randomize the value of the array
        _array1 = GenerateRandomArray(arraySize);
        _array2 = GenerateRandomArray(arraySize);
        // send the array to gpu
        _buffer1.SetData(_array1);
        _buffer2.SetData(_array2);
        // initialize the array of result
        _resultArray = new float[arraySize];
    }

    /// <summary>
    ///     Initializes the compute buffers.
    /// </summary>
    /// <param name="arraySize">The size of the buffers to initialize.</param>
    private void InitializeBuffers(int arraySize)
    {
        // initialize the buffer to structured default type
        _buffer1 = new ComputeBuffer(arraySize, sizeof(float));
        _buffer2 = new ComputeBuffer(arraySize, sizeof(float));
        _resultBuffer = new ComputeBuffer(arraySize, sizeof(float));
    }

    /// <summary>
    ///     Releases the compute buffers.
    /// </summary>
    private void ReleaseBuffers()
    {
        // release the memory (for cuda and on unity side)
        _vectorAddCuda.DestroyActionsAdd();
        _buffer1.Release();
        _buffer2.Release();
        _resultBuffer.Release();
    }

    /// <summary>
    ///     Generates a random array of floats. (with uniform law between 0 and 100)
    /// </summary>
    /// <param name="length">The length of the array to generate.</param>
    /// <returns>A randomly generated float array.</returns>
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
    ///     Computes the sum of two arrays using the CPU.
    /// </summary>
    /// <returns>The resulting array after summing the input arrays.</returns>
    private float[] ComputeCPUSum()
    {
        float[] result = new float[_array1.Length];
        for (int i = 0; i < _array1.Length; i++)
        {
            result[i] = _array1[i] + _array2[i];
        }

        return result;
    }

    /// <summary>
    ///     Compares the GPU result with the CPU result.
    /// </summary>
    /// <param name="cpuResult">The result from the CPU computation.</param>
    /// <returns>True if the results are equal, false otherwise.</returns>
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

    /// <summary>
    ///     Updates the profiling information and displays it.
    /// </summary>
    /// <param name="executionTimeCS">The execution time for the compute shader.</param>
    /// <param name="executionTimeCUDA">The execution time for the CUDA implementation.</param>
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

    /// <summary>
    ///     Logs the profiling summary for the current array size.
    /// </summary>
    private void LogProfilingSummary()
    {
        if (_executionCount > 0)
        {
            // Calculate overall averages
            float overallAverageCS = _totalExecutionTimeCS / _executionCount;
            float overallAverageCUDA = _totalExecutionTimeCUDA / _executionCount;

            // Calculate variance and standard deviation for Compute Shader (CS)
            float sumOfSquaredDifferencesCS = 0f;
            foreach (float time in _executionTimesCS)
            {
                float difference = time - overallAverageCS;
                sumOfSquaredDifferencesCS += difference * difference;
            }

            float varianceCS = sumOfSquaredDifferencesCS / _executionCount;
            float standardDeviationCS = Mathf.Sqrt(varianceCS);

            // Calculate variance and standard deviation for CUDA
            float sumOfSquaredDifferencesCUDA = 0f;
            foreach (float time in _executionTimesCUDA)
            {
                float difference = time - overallAverageCUDA;
                sumOfSquaredDifferencesCUDA += difference * difference;
            }

            float varianceCUDA = sumOfSquaredDifferencesCUDA / _executionCount;
            float standardDeviationCUDA = Mathf.Sqrt(varianceCUDA);

            // Log the results for Compute Shader (CS)
            Debug.Log($"Compute Shader Profiling Summary for Array Size {_arraySizes[_currentArraySizeIndex]}:" +
                      $" Samples: {_executionCount}" +
                      $", Average Execution Time (CS): {overallAverageCS:F3} ms" +
                      $", Standard Deviation (CS): {standardDeviationCS:F3} ms" +
                      $", Variance (CS): {varianceCS:F3} ms^2");

            // Log the results for CUDA
            Debug.Log($"CUDA Profiling Summary for Array Size {_arraySizes[_currentArraySizeIndex]}:" +
                      $" Samples: {_executionCount}" +
                      $", Average Execution Time (CUDA): {overallAverageCUDA:F3} ms" +
                      $", Standard Deviation (CUDA): {standardDeviationCUDA:F3} ms" +
                      $", Variance (CUDA): {varianceCUDA:F3} ms^2");

            // Store the results in the profiling data list
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

    /// <summary>
    ///     Exports the profiling data to a CSV file.
    /// </summary>
    private void ExportToCsv()
    {
        string filePath = Path.Combine(Application.dataPath, "ProfilingResults.csv");
        using (StreamWriter writer = new(filePath))
        {
            // Write CSV header
            writer.WriteLine(
                "ArraySize;SampleCount;AverageExecutionTimeCS;StandardDeviationCS;VarianceCS;AverageExecutionTimeCUDA;StandardDeviationCUDA;VarianceCUDA;Difference");

            // Write each profiling data entry
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

/// <summary>
///     Class to store profiling data for each test run.
/// </summary>
public class ProfilingDataSum
{
    public int ArraySize { get; set; } // The size of the array used in the test

    public int SampleCount { get; set; } // The number of samples taken

    public float AverageExecutionTimeCS { get; set; } // Average execution time for Compute Shader (CS)

    public float StandardDeviationCS { get; set; } // Standard deviation for CS execution time

    public float VarianceCS { get; set; } // Variance for CS execution time

    public float AverageExecutionTimeCUDA { get; set; } // Average execution time for CUDA

    public float StandardDeviationCUDA { get; set; } // Standard deviation for CUDA execution time

    public float VarianceCUDA { get; set; } // Variance for CUDA execution time
}

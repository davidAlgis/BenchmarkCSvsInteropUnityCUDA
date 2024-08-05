using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
///     This class manages the vector addition process and profiling.
/// </summary>
public class BenchmarkManager : MonoBehaviour
{
    // Configuration parameters for array sizes and samples per size
    [SerializeField] protected List<int> _arraySizes = new()
    {
        1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000, 2500000, 5000000,
        7500000, 10000000
    };

    [SerializeField] protected int _numSamplesPerSize = 10000;

    // UI components for displaying profiling information
    // Reference to TextMeshPro component for displaying CS profiling info
    [SerializeField] private TextMeshProUGUI _profileTextCS;

    // Reference to TextMeshPro component for displaying CUDA profiling info
    [SerializeField] private TextMeshProUGUI _profileTextCUDA;

    // Reference to TextMeshPro component for displaying difference info
    [SerializeField] private TextMeshProUGUI _profileTextDifference;

    // Reference to TextMeshPro component for displaying array size
    [SerializeField] protected TextMeshProUGUI _titleText;

    // Time to wait before recording profiling data
    [SerializeField] private float _waitTimeBeforeClocking = 5f;

    // Lists to store execution times and profiling results
    private readonly List<float> _executionTimesCS = new();
    private readonly List<float> _executionTimesCUDA = new();

    private readonly List<ProfilingDataSum> _profilingResults = new();

    // Indices and counters for tracking progress
    protected int _currentArraySizeIndex;

    protected int _currentSampleCount;
    private int _executionCount;

    private float _maxExecutionTimeCS = float.MinValue;
    private float _maxExecutionTimeCUDA = float.MinValue;

    // Minimum and maximum execution times
    private float _minExecutionTimeCS = float.MaxValue;
    private float _minExecutionTimeCUDA = float.MaxValue;
    private bool _recordingStarted;

    private float _startTime;
    private float _totalExecutionTimeCS;
    private float _totalExecutionTimeCUDA;

    /// <summary>
    ///     Initializes the components and starts the profiling process.
    /// </summary>
    protected virtual void Start()
    {
        _startTime = Time.time;
        _recordingStarted = false;
        Initialize();
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
            UpdateBeforeRecord();
        }

        if (_recordingStarted && _currentSampleCount < _numSamplesPerSize)
        {
            UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA);

            UpdateProfilingInfo(gpuExecutionTimeCS, gpuExecutionTimeCUDA);
            // Update the UI
            _currentSampleCount++;
        }
        // When we reach the number of sample required
        else if (_currentSampleCount >= _numSamplesPerSize)
        {
            UpdatePostSample();
        }
    }

    protected virtual void Initialize() { }

    protected virtual void UpdateBeforeRecord()
    {
        _profileTextCS.text = "Average Time CS: loading...";
        _profileTextCUDA.text = "Average Time CUDA: loading...";
        _profileTextDifference.text = "Difference: loading...";

        // launch record when we reach the _timeAfterRecord
        if (Time.time - _startTime >= _waitTimeBeforeClocking)
        {
            _recordingStarted = true;
        }
    }

    protected virtual void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA)
    {
        gpuExecutionTimeCS = 0.0f;
        gpuExecutionTimeCUDA = 0.0f;
    }

    protected virtual void UpdatePostSample()
    {
        // log the data associated to the test
        LogProfilingSummary();

        _currentSampleCount = 0;
        _executionTimesCS.Clear();
        _executionTimesCUDA.Clear();
        _totalExecutionTimeCS = 0;
        _totalExecutionTimeCUDA = 0;
        _executionCount = 0;
        _minExecutionTimeCS = float.MaxValue;
        _maxExecutionTimeCS = float.MinValue;
        _minExecutionTimeCUDA = float.MaxValue;
        _maxExecutionTimeCUDA = float.MinValue;

        _currentArraySizeIndex++;
        // if we are not at the end of the array of size, we re-initialize the array with the new size of array
        if (_currentArraySizeIndex < _arraySizes.Count)
        {
            ReInitialize();
        }
        else
        {
            EndTest();
        }
    }

    protected virtual void ReInitialize() { }

    /// <summary>
    ///     Ends the current test. If there are more scenes in the build settings, it will load the next scene.
    ///     Otherwise, it will export the profiling data to a CSV file and quit the application.
    /// </summary>
    private void EndTest()
    {
        // if we are at the end we export all the data to a csv and stop the application
        Debug.Log("All tests completed. Export to csv the result.");
        ExportToCsv();

        // Get the current active scene's build index
        int currentSceneIndex = SceneManager.GetActiveScene().buildIndex;

        // Check if there is a next scene in the build settings
        if (currentSceneIndex + 1 < SceneManager.sceneCountInBuildSettings)
        {
            // Load the next scene
            SceneManager.LoadScene(currentSceneIndex + 1);
        }
        else
        {
            // No more scenes to load, quit the application
#if UNITY_EDITOR
            EditorApplication.isPlaying = false;
#elif DEVELOPMENT_BUILD
        Application.Quit();
#endif
        }
    }

    /// <summary>
    ///     Generates a random array of floats. (with uniform law between 0 and 100)
    /// </summary>
    /// <param name="length">The length of the array to generate.</param>
    /// <returns>A randomly generated float array.</returns>
    protected float[] GenerateRandomArray(int length, float min = 0.0f, float max = 100.0f)
    {
        float[] array = new float[length];
        for (int i = 0; i < length; i++)
        {
            // array[i] = Random.Range(min, max);
            array[i] = 1f;
        }

        return array;
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

        // Update min and max execution times for CS
        if (executionTimeCS < _minExecutionTimeCS)
        {
            _minExecutionTimeCS = executionTimeCS;
        }

        if (executionTimeCS > _maxExecutionTimeCS)
        {
            _maxExecutionTimeCS = executionTimeCS;
        }

        // Update min and max execution times for CUDA
        if (executionTimeCUDA < _minExecutionTimeCUDA)
        {
            _minExecutionTimeCUDA = executionTimeCUDA;
        }

        if (executionTimeCUDA > _maxExecutionTimeCUDA)
        {
            _maxExecutionTimeCUDA = executionTimeCUDA;
        }

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
                      $", Min Execution Time (CS): {_minExecutionTimeCS:F3} ms" +
                      $", Max Execution Time (CS): {_maxExecutionTimeCS:F3} ms" +
                      $", Standard Deviation (CS): {standardDeviationCS:F3} ms" +
                      $", Variance (CS): {varianceCS:F3} ms^2");

            // Log the results for CUDA
            Debug.Log($"CUDA Profiling Summary for Array Size {_arraySizes[_currentArraySizeIndex]}:" +
                      $" Samples: {_executionCount}" +
                      $", Average Execution Time (CUDA): {overallAverageCUDA:F3} ms" +
                      $", Min Execution Time (CUDA): {_minExecutionTimeCUDA:F3} ms" +
                      $", Max Execution Time (CUDA): {_maxExecutionTimeCUDA:F3} ms" +
                      $", Standard Deviation (CUDA): {standardDeviationCUDA:F3} ms" +
                      $", Variance (CUDA): {varianceCUDA:F3} ms^2");

            // Store the results in the profiling data list
            _profilingResults.Add(new ProfilingDataSum
            {
                ArraySize = _arraySizes[_currentArraySizeIndex],
                AverageExecutionTimeCS = overallAverageCS,
                StandardDeviationCS = standardDeviationCS,
                VarianceCS = varianceCS,
                MinExecutionTimeCS = _minExecutionTimeCS,
                MaxExecutionTimeCS = _maxExecutionTimeCS,
                AverageExecutionTimeCUDA = overallAverageCUDA,
                StandardDeviationCUDA = standardDeviationCUDA,
                VarianceCUDA = varianceCUDA,
                MinExecutionTimeCUDA = _minExecutionTimeCUDA,
                MaxExecutionTimeCUDA = _maxExecutionTimeCUDA,
                SampleCount = _executionCount
            });
        }
    }

    /// <summary>
    ///     Exports the profiling data to a CSV file.
    /// </summary>
    private void ExportToCsv()
    {
        string sceneName = SceneManager.GetActiveScene().name;
        string graphicsAPI = SystemInfo.graphicsDeviceType.ToString();
        string fileName = $"ProfilingResults - {sceneName} - {_numSamplesPerSize} - {graphicsAPI}.csv";
        string filePath = Path.Combine(Application.dataPath, fileName);

        using (StreamWriter writer = new(filePath))
        {
            // Write CSV header
            writer.WriteLine(
                "ArraySize;SampleCount;AverageExecutionTimeCS;MinExecutionTimeCS;MaxExecutionTimeCS;StandardDeviationCS;VarianceCS;AverageExecutionTimeCUDA;MinExecutionTimeCUDA;MaxExecutionTimeCUDA;StandardDeviationCUDA;VarianceCUDA;Difference");

            // Write each profiling data entry
            foreach (ProfilingDataSum data in _profilingResults)
            {
                float difference = data.AverageExecutionTimeCS - data.AverageExecutionTimeCUDA;
                writer.WriteLine(
                    $"{data.ArraySize};{data.SampleCount};{data.AverageExecutionTimeCS:F3};{data.MinExecutionTimeCS:F3};{data.MaxExecutionTimeCS:F3};{data.StandardDeviationCS:F3};{data.VarianceCS:F3};{data.AverageExecutionTimeCUDA:F3};{data.MinExecutionTimeCUDA:F3};{data.MaxExecutionTimeCUDA:F3};{data.StandardDeviationCUDA:F3};{data.VarianceCUDA:F3};{difference:F3}");
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

    public float MinExecutionTimeCS { get; set; } // Minimum execution time for Compute Shader (CS)

    public float MaxExecutionTimeCS { get; set; } // Maximum execution time for Compute Shader (CS)

    public float StandardDeviationCS { get; set; } // Standard deviation for CS execution time

    public float VarianceCS { get; set; } // Variance for CS execution time

    public float AverageExecutionTimeCUDA { get; set; } // Average execution time for CUDA

    public float MinExecutionTimeCUDA { get; set; } // Minimum execution time for CUDA

    public float MaxExecutionTimeCUDA { get; set; } // Maximum execution time for CUDA

    public float StandardDeviationCUDA { get; set; } // Standard deviation for CUDA execution time

    public float VarianceCUDA { get; set; } // Variance for CUDA execution time
}

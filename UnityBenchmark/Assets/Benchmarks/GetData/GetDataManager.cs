using System.Diagnostics;
using UnityEngine;

/// <summary>
///     This class manages the copy from GPU to CPU of a buffer process and profiling.
/// </summary>
public class GetDataManager : BenchmarkManager
{
    // References to CUDA class
    [SerializeField] private GetDataCUDA _getDataCuda;

    // Arrays for computation
    private float[] _array;

    // Compute buffers for GPU computation
    private ComputeBuffer _bufferCS;
    private ComputeBuffer _bufferCUDA;
    private bool _hasBeenRelease;

    private bool _randomizedEachFrame;

    /// <summary>
    ///     Initializes the components and starts the profiling process.
    /// </summary>
    protected override void Start()
    {
        _getDataCuda.InitializeInteropHandler();
        base.Start();
    }

    private void OnDestroy()
    {
        if (!_hasBeenRelease)
        {
            ReleaseBuffers();
        }
    }

    /// <summary>
    ///     Initializes the components specific to vector addition.
    /// </summary>
    protected override void Initialize()
    {
        base.Initialize();
        InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
        _getDataCuda.InitializeActionsGetData(_arraySizes[_currentArraySizeIndex], _bufferCUDA);

        // Execute CUDA get data once to get an initial execution time
        _getDataCuda.UpdateGetData();
    }

    /// <summary>
    ///     Updates the profiling process before recording starts.
    /// </summary>
    protected override void UpdateBeforeRecord()
    {
        _bufferCS.SetData(_array);
        _bufferCUDA.SetData(_array);
        _getDataCuda.UpdateGetData();

        base.UpdateBeforeRecord();
    }

    /// <summary>
    ///     Updates the main recording process.
    /// </summary>
    protected override void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA,
        out float cpuExecutionTimeCPU)
    {
        int arraySize = _arraySizes[_currentArraySizeIndex];
        _titleText.text = $"Get Data - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";

        if (_randomizedEachFrame)
        {
            InitializeArrays(arraySize);
        }

        // Retrieve the data from the result buffer
        gpuExecutionTimeCS = UpdateGetDataCS();
        gpuExecutionTimeCUDA = _getDataCuda.UpdateGetData();
        cpuExecutionTimeCPU = 0.0f;
    }

    private float UpdateGetDataCS()
    {
        // We call GetData to make a first synchronization before chrono and to make sure that GPU and CPU are fully
        // synchronize and that the chrono retrieve only the correct time and not other GPU execution time.
        _bufferCS.GetData(_array, 0, 0, 1);
        // Start the stopwatch
        Stopwatch stopwatch = Stopwatch.StartNew();

        // Retrieve the data from the result buffer
        _bufferCS.GetData(_array);

        // Stop the stopwatch and return the elapsed time
        stopwatch.Stop();
        return (float)stopwatch.Elapsed.TotalMilliseconds;
    }

    /// <summary>
    ///     Re-initializes the components for the next array size.
    /// </summary>
    protected override void ReInitialize()
    {
        InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
        _getDataCuda.InitializeActionsGetData(_arraySizes[_currentArraySizeIndex], _bufferCUDA);
        _getDataCuda.UpdateGetData();
    }

    /// <summary>
    ///     Initializes the arrays for computation.
    /// </summary>
    /// <param name="arraySize">The size of the arrays to initialize.</param>
    private void InitializeArrays(int arraySize)
    {
        _array = GenerateRandomArray(arraySize);
        _bufferCUDA.SetData(_array);
        _bufferCS.SetData(_array);
    }

    /// <summary>
    ///     Initializes the compute buffers.
    /// </summary>
    /// <param name="arraySize">The size of the buffers to initialize.</param>
    private void InitializeBuffers(int arraySize)
    {
        _hasBeenRelease = false;
        _bufferCUDA = new ComputeBuffer(arraySize, sizeof(float));
        _bufferCS = new ComputeBuffer(arraySize, sizeof(float));
    }

    /// <summary>
    ///     Releases the compute buffers.
    /// </summary>
    private void ReleaseBuffers()
    {
        _hasBeenRelease = true;
        _getDataCuda.DestroyActionsGetData();
        _bufferCUDA.Release();
        _bufferCS.Release();
    }
}

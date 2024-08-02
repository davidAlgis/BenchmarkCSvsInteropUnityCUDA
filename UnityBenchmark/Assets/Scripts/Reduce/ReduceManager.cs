using UnityEngine;

/// <summary>
///     This class manages the reduction of a buffer process and profiling.
/// </summary>
public class ReduceManager : BenchmarkManager
{
    [SerializeField] private ComputeShader _computeShader;

    // References to CUDA class
    [SerializeField] private ReduceCUDA _reduceCuda;

    // Arrays for computation
    private float[] _array;

    // Compute buffers for GPU computation
    private ComputeBuffer _bufferCS;
    private ComputeBuffer _bufferCUDA;

    private bool _hasBeenRelease;
    private bool _randomizedEachFrame;

    // Compute shader for reduction
    private ReduceCS _reduceCS;
    private float[] _resultArray;
    private ComputeBuffer _resultBufferCS;
    private ComputeBuffer _spinlockBuffer;

    /// <summary>
    ///     Initializes the components and starts the profiling process.
    /// </summary>
    protected override void Start()
    {
        _reduceCuda.InitializeInteropHandler();
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
        InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
        _reduceCuda.InitializeActionsReduce(_arraySizes[_currentArraySizeIndex], _bufferCUDA);

        // Initialize the reduce compute shader
        _reduceCS = new ReduceCS();
        _reduceCS.Init(_computeShader, _arraySizes[_currentArraySizeIndex], _bufferCS, _resultBufferCS,
            _spinlockBuffer);

        // Execute CUDA get data once to get an initial execution time
        _reduceCuda.UpdateReduce();
    }

    /// <summary>
    ///     Updates the profiling process before recording starts.
    /// </summary>
    protected override void UpdateBeforeRecord()
    {
        _bufferCS.SetData(_array);
        _bufferCUDA.SetData(_array);
        _reduceCuda.UpdateReduce();

        base.UpdateBeforeRecord();
    }

    /// <summary>
    ///     Updates the main recording process.
    /// </summary>
    protected override void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA)
    {
        int arraySize = _arraySizes[_currentArraySizeIndex];
        _titleText.text = $"Reduce - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";

        if (_randomizedEachFrame)
        {
            InitializeArrays(arraySize);
        }

        _spinlockBuffer.SetData(new[] { 0 });
        _resultBufferCS.SetData(new[] { 0 });
        // Perform reduction using compute shader
        gpuExecutionTimeCS = _reduceCS.ComputeSum(_resultBufferCS, ref _resultArray);

        // Perform reduction using CUDA
        gpuExecutionTimeCUDA = _reduceCuda.UpdateReduce();
    }

    /// <summary>
    ///     Re-initializes the components for the next array size.
    /// </summary>
    protected override void ReInitialize()
    {
        InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
        _reduceCuda.InitializeActionsReduce(_arraySizes[_currentArraySizeIndex], _bufferCUDA);

        // Re-initialize the reduce compute shader
        _reduceCS.Init(_computeShader, _arraySizes[_currentArraySizeIndex], _bufferCS, _resultBufferCS,
            _spinlockBuffer);
        _reduceCuda.UpdateReduce();
    }

    /// <summary>
    ///     Initializes the arrays for computation.
    /// </summary>
    /// <param name="arraySize">The size of the arrays to initialize.</param>
    private void InitializeArrays(int arraySize)
    {
        _array = GenerateRandomArray(arraySize, -10.0f, 10.0f);
        _resultArray = new float[1];
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
        _resultBufferCS = new ComputeBuffer(1, sizeof(float));
        _spinlockBuffer = new ComputeBuffer(1, sizeof(int));
        _spinlockBuffer.SetData(new[] { 0 }); // Initialize spinlock to 0
    }

    /// <summary>
    ///     Releases the compute buffers.
    /// </summary>
    private void ReleaseBuffers()
    {
        _hasBeenRelease = true;
        _reduceCuda.DestroyActionsReduce();
        _bufferCUDA.Release();
        _bufferCS.Release();
        _resultBufferCS.Release();
        _spinlockBuffer.Release();
    }
}

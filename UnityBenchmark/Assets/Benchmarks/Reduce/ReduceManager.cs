using UnityEngine;

/// <summary>
///     This class manages the reduction of a buffer process and profiling.
/// </summary>
public class ReduceManager : BenchmarkManager
{
    [SerializeField] private ComputeShader _computeShader;

    // References to CUDA class
    [SerializeField] private ReduceCUDA _reduceCuda;

    // Boolean to check the result each frame
    [SerializeField] private bool _checkResultEachFrame;

    // Arrays for computation
    private float[] _array;

    // Compute buffers for GPU computation
    private ComputeBuffer _bufferCS;
    private ComputeBuffer _bufferCUDA;

    private bool _hasBeenRelease;
    private bool _randomizedEachFrame;

    // CPU reduction instance
    private ReduceCPU _reduceCPU;

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
        base.Initialize();
        InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
        _reduceCuda.InitializeActionsReduce(_arraySizes[_currentArraySizeIndex], _bufferCUDA);

        // Initialize the reduce compute shader
        _reduceCS = new ReduceCS();
        _reduceCS.Init(_computeShader, _arraySizes[_currentArraySizeIndex], _bufferCS, _resultBufferCS,
            _spinlockBuffer);

        // Initialize the CPU reduction
        _reduceCPU = new ReduceCPU();
        _reduceCPU.Init(_arraySizes[_currentArraySizeIndex]);
        // Copy the generated array to the CPU reduction instance for fair comparison
        _reduceCPU.SetData(_array);

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
    protected override void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA,
        out float cpuExecutionTime)
    {
        int arraySize = _arraySizes[_currentArraySizeIndex];
        _titleText.text = $"Reduce - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";

        if (_randomizedEachFrame)
        {
            InitializeArrays(arraySize);
        }

        // _spinlockBuffer.SetData(new[] { 0 });
        _resultBufferCS.SetData(new[] { 0 });

        // Perform reduction using compute shader
        gpuExecutionTimeCS = _reduceCS.ComputeSum(_resultBufferCS, arraySize, ref _resultArray);

        // Perform reduction using CUDA
        gpuExecutionTimeCUDA = _reduceCuda.UpdateReduce();
        float cpuSum = 0.0f;

        cpuExecutionTime = _isCPUTimeTooLarge
            ? 0.0f
            :
            // Perform reduction on the CPU
            cpuExecutionTime = _reduceCPU.ComputeSum(out cpuSum);

        // Check result each frame if enabled
        if (_checkResultEachFrame)
        {
            CheckResult(cpuSum);
        }
    }

    /// <summary>
    ///     Re-initializes the components for the next array size.
    /// </summary>
    protected override void ReInitialize()
    {
        ReleaseBuffers();
        InitializeBuffers(_arraySizes[_currentArraySizeIndex]);
        InitializeArrays(_arraySizes[_currentArraySizeIndex]);
        _reduceCuda.InitializeActionsReduce(_arraySizes[_currentArraySizeIndex], _bufferCUDA);

        // Re-initialize the reduce compute shader
        _reduceCS.Init(_computeShader, _arraySizes[_currentArraySizeIndex], _bufferCS, _resultBufferCS,
            _spinlockBuffer);

        // Re-initialize the CPU reduction
        _reduceCPU.Init(_arraySizes[_currentArraySizeIndex]);

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
        _spinlockBuffer.SetData(new[] { 0 });
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

    /// <summary>
    ///     Checks if the result of the reduction is correct by comparing it to a CPU sum.
    /// </summary>
    private void CheckResult(float cpuSum)
    {
        // Get the GPU result from the last compute shader run
        float gpuSum = _resultArray[0];

        // Compare CPU and GPU results
        if (Mathf.Approximately(cpuSum, gpuSum))
        {
            Debug.Log("GPU result matches CPU result.");
        }
        else
        {
            Debug.LogError($"GPU result ({gpuSum}) does not match CPU result ({cpuSum}).");
        }
    }
}

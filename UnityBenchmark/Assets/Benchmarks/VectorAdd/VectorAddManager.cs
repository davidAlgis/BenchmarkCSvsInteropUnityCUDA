using UnityEngine;

/// <summary>
///     This class manages the vector addition process and profiling.
/// </summary>
public class VectorAddManager : BenchmarkManager
{
    // References to compute shader and CUDA class
    [SerializeField] private ComputeShader _computeShader;
    [SerializeField] private VectorAddCUDA _vectorAddCuda;

    [SerializeField] private int _nbrElementToRetrieve = 1;

    // Arrays for computation
    private float[] _array1;
    private float[] _array2;

    // Compute buffers for GPU computation
    private ComputeBuffer _buffer1CS;
    private ComputeBuffer _buffer2CS;
    private ComputeBuffer _resultBufferCS;

    private ComputeBuffer _buffer1CUDA;
    private ComputeBuffer _buffer2CUDA;
    private ComputeBuffer _resultBufferCUDA;

    private bool _compareCPU;
    private bool _hasBeenRelease;

    private bool _randomizedEachFrame;

    // Arrays for storing results
    private float[] _resultArrayCS;
    private float[] _resultArrayCUDA;

    // Compute shader class for vector addition
    private VectorAddCS _vectorAddCompute;

    // CPU reduction instance
    private VectorAddCPU _vectorAddCPU;

    /// <summary>
    ///     Initializes the components and starts the profiling process.
    /// </summary>
    protected override void Start()
    {
        _vectorAddCuda.InitializeInteropHandler();
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
        _vectorAddCompute = new VectorAddCS();
        _vectorAddCPU = new VectorAddCPU(); // Initialize CPU vector add instance

        int currentArraySize = _arraySizes[_currentArraySizeIndex];
        InitializeBuffers(currentArraySize);
        InitializeArrays(currentArraySize);

        // Initialize Compute Shader for vector addition
        _vectorAddCompute.Init(_computeShader, currentArraySize, _buffer1CS, _buffer2CS, _resultBufferCS);

        // Initialize CUDA for vector addition
        _vectorAddCuda.InitializeActionsAdd(currentArraySize, _buffer1CUDA, _buffer2CUDA, _resultBufferCUDA,
            _nbrElementToRetrieve);

        // Initialize CPU vector addition
        _vectorAddCPU.Init(currentArraySize);

        // Execute CUDA vector addition once to get an initial execution time
        _vectorAddCuda.ComputeSum();
    }

    /// <summary>
    ///     Updates the profiling process before recording starts.
    /// </summary>
    protected override void UpdateBeforeRecord()
    {
        _buffer1CS.SetData(_array1);
        _buffer2CS.SetData(_array2);
        _buffer1CUDA.SetData(_array1);
        _buffer2CUDA.SetData(_array2);

        // Perform initial computations to stabilize timings
        _vectorAddCompute.ComputeSum(_resultBufferCS, _arraySizes[_currentArraySizeIndex], ref _resultArrayCS,
            _nbrElementToRetrieve);
        _vectorAddCuda.ComputeSum();
        _vectorAddCPU.ComputeSum(out _); // Execute CPU sum without capturing the result

        base.UpdateBeforeRecord();
    }

    /// <summary>
    ///     Records profiling data for the main computational tasks and calls the main computational task for compute shader, CUDA, and CPU.
    /// </summary>
    /// <param name="gpuExecutionTimeCS">The execution time for the compute shader.</param>
    /// <param name="gpuExecutionTimeCUDA">The execution time for the CUDA implementation.</param>
    /// <param name="cpuExecutionTime">The execution time for the CPU implementation.</param>
    protected override void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA,
        out float cpuExecutionTime)
    {
        int arraySize = _arraySizes[_currentArraySizeIndex];
        _titleText.text = $"Vector Add - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";

        if (_randomizedEachFrame)
        {
            InitializeArrays(arraySize);
        }

        // Perform reduction using compute shader
        gpuExecutionTimeCS =
            _vectorAddCompute.ComputeSum(_resultBufferCS, arraySize, ref _resultArrayCS, _nbrElementToRetrieve);

        // Perform reduction using CUDA
        gpuExecutionTimeCUDA = _vectorAddCuda.ComputeSum();

        // Perform reduction on the CPU
        cpuExecutionTime = _vectorAddCPU.ComputeSum(out float[] cpuResult);

        // Check result each frame if enabled
        if (_compareCPU)
        {
            bool isEqual = CompareResults(cpuResult);
            if (!isEqual)
            {
                Debug.LogError("GPU and CPU results do not match!");
            }
        }
    }

    /// <summary>
    ///     Re-initializes the components for the next array size.
    /// </summary>
    protected override void ReInitialize()
    {
        ReleaseBuffers();
        int currentArraySize = _arraySizes[_currentArraySizeIndex];
        InitializeBuffers(currentArraySize);
        InitializeArrays(currentArraySize);

        // Re-initialize Compute Shader for vector addition
        _vectorAddCompute.Init(_computeShader, currentArraySize, _buffer1CS, _buffer2CS, _resultBufferCS);

        // Re-initialize CUDA for vector addition
        _vectorAddCuda.InitializeActionsAdd(currentArraySize, _buffer1CUDA, _buffer2CUDA, _resultBufferCUDA,
            _nbrElementToRetrieve);
        _vectorAddCuda.ComputeSum();

        // Re-initialize CPU vector addition
        _vectorAddCPU.Init(currentArraySize);
    }

    /// <summary>
    ///     Initializes the arrays for computation.
    /// </summary>
    /// <param name="arraySize">The size of the arrays to initialize.</param>
    private void InitializeArrays(int arraySize)
    {
        _array1 = GenerateRandomArray(arraySize, -10.0f, 10.0f);
        _array2 = GenerateRandomArray(arraySize, -10.0f, 10.0f);

        // Set data for Compute Shader
        _buffer1CS.SetData(_array1);
        _buffer2CS.SetData(_array2);

        // Set data for CUDA
        _buffer1CUDA.SetData(_array1);
        _buffer2CUDA.SetData(_array2);

        _resultArrayCS = new float[_nbrElementToRetrieve];
        _resultArrayCUDA = new float[_nbrElementToRetrieve];
    }

    /// <summary>
    ///     Initializes the compute buffers.
    /// </summary>
    /// <param name="arraySize">The size of the buffers to initialize.</param>
    private void InitializeBuffers(int arraySize)
    {
        _hasBeenRelease = false;

        // Initialize Compute Shader buffers
        _buffer1CS = new ComputeBuffer(arraySize, sizeof(float));
        _buffer2CS = new ComputeBuffer(arraySize, sizeof(float));
        _resultBufferCS = new ComputeBuffer(_nbrElementToRetrieve, sizeof(float));

        // Initialize CUDA buffers
        _buffer1CUDA = new ComputeBuffer(arraySize, sizeof(float));
        _buffer2CUDA = new ComputeBuffer(arraySize, sizeof(float));
        _resultBufferCUDA = new ComputeBuffer(_nbrElementToRetrieve, sizeof(float));
    }

    /// <summary>
    ///     Releases the compute buffers.
    /// </summary>
    private void ReleaseBuffers()
    {
        _hasBeenRelease = true;

        // Release Compute Shader buffers
        if (_buffer1CS != null)
        {
            _buffer1CS.Release();
        }

        if (_buffer2CS != null)
        {
            _buffer2CS.Release();
        }

        if (_resultBufferCS != null)
        {
            _resultBufferCS.Release();
        }

        // Release CUDA buffers
        if (_buffer1CUDA != null)
        {
            _buffer1CUDA.Release();
        }

        if (_buffer2CUDA != null)
        {
            _buffer2CUDA.Release();
        }

        if (_resultBufferCUDA != null)
        {
            _resultBufferCUDA.Release();
        }

        // Destroy CUDA actions
        _vectorAddCuda.DestroyActionsAdd();
    }

    /// <summary>
    ///     Compares the GPU result with the CPU result.
    /// </summary>
    /// <param name="cpuResult">The result from the CPU computation.</param>
    /// <returns>True if the results are equal within a small epsilon, false otherwise.</returns>
    private bool CompareResults(float[] cpuResult)
    {
        bool isEqual = true;
        for (int i = 0; i < _nbrElementToRetrieve; i++)
        {
            if (Mathf.Abs(_resultArrayCS[i] - cpuResult[i]) > Mathf.Epsilon)
            {
                isEqual = false;
                Debug.LogError($"Mismatch at index {i}: GPU = {_resultArrayCS[i]}, CPU = {cpuResult[i]}");
            }
        }

        return isEqual;
    }
}

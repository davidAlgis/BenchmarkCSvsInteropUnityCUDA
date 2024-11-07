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

    private ComputeBuffer _buffer1CUDA;
    private ComputeBuffer _buffer2CS;
    private ComputeBuffer _buffer2CUDA;
    private bool _compareCPU;
    private bool _hasBeenRelease;

    private bool _randomizedEachFrame;

    // Arrays for storing results
    private float[] _resultArrayCS;
    private ComputeBuffer _resultBufferCS;
    private ComputeBuffer _resultBufferCUDA;

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
        _vectorAddCPU = new VectorAddCPU();
        int arraySize = _arraySizes[_currentArraySizeIndex];
        InitializeBuffers(arraySize);
        InitializeArrays(arraySize);
        _vectorAddCompute.Init(_computeShader, arraySize, _buffer1CS, _buffer2CS,
            _resultBufferCS);
        _vectorAddCuda.InitializeActionsAdd(arraySize, _buffer1CUDA, _buffer2CUDA,
            _resultBufferCUDA, _nbrElementToRetrieve);
        // Initialize CPU vector addition
        _vectorAddCPU.Init(arraySize);

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
        _vectorAddCompute.ComputeSum(_resultBufferCS, _arraySizes[_currentArraySizeIndex], ref _resultArrayCS,
            _nbrElementToRetrieve);
        _vectorAddCuda.ComputeSum();
        _vectorAddCPU.ComputeSum(out _);

        base.UpdateBeforeRecord();
    }

    /// <summary>
    ///     Updates the main recording process.
    /// </summary>
    protected override void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA, out float cpuExecutionTime)
    {
        int arraySize = _arraySizes[_currentArraySizeIndex];
        _titleText.text = $"Vector Add - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";

        if (_randomizedEachFrame)
        {
            InitializeArrays(arraySize);
        }

        gpuExecutionTimeCS =
            _vectorAddCompute.ComputeSum(_resultBufferCS, arraySize, ref _resultArrayCS, _nbrElementToRetrieve);
        gpuExecutionTimeCUDA = _vectorAddCuda.ComputeSum();
        // Perform reduction on the CPU
        cpuExecutionTime = _vectorAddCPU.ComputeSum(out float[] cpuResult);

        if (_compareCPU)
        {
            float[] cpuSum = ComputeCPUSum();
            if (!CompareResults(cpuSum))
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
        int arraySize = _arraySizes[_currentArraySizeIndex];
        InitializeBuffers(arraySize);
        InitializeArrays(arraySize);
        _vectorAddCompute.Init(_computeShader, arraySize, _buffer1CS, _buffer2CS,
            _resultBufferCS);
        _vectorAddCuda.InitializeActionsAdd(arraySize, _buffer1CUDA, _buffer2CUDA,
            _resultBufferCUDA, _nbrElementToRetrieve);
        _vectorAddCuda.ComputeSum();
        // Re-initialize CPU vector addition
        _vectorAddCPU.Init(arraySize);
    }

    /// <summary>
    ///     Initializes the arrays for computation.
    /// </summary>
    /// <param name="arraySize">The size of the arrays to initialize.</param>
    private void InitializeArrays(int arraySize)
    {
        _array1 = GenerateRandomArray(arraySize);
        _array2 = GenerateRandomArray(arraySize);
        _buffer1CS.SetData(_array1);
        _buffer2CS.SetData(_array2);
        _buffer1CUDA.SetData(_array1);
        _buffer2CUDA.SetData(_array2);
        _resultArrayCS = new float[_nbrElementToRetrieve];
    }

    /// <summary>
    ///     Initializes the compute buffers.
    /// </summary>
    /// <param name="arraySize">The size of the buffers to initialize.</param>
    private void InitializeBuffers(int arraySize)
    {
        _hasBeenRelease = false;
        _buffer1CS = new ComputeBuffer(arraySize, sizeof(float));
        _buffer2CS = new ComputeBuffer(arraySize, sizeof(float));
        _resultBufferCS = new ComputeBuffer(arraySize, sizeof(float));

        _buffer1CUDA = new ComputeBuffer(arraySize, sizeof(float));
        _buffer2CUDA = new ComputeBuffer(arraySize, sizeof(float));
        _resultBufferCUDA = new ComputeBuffer(arraySize, sizeof(float));
    }

    /// <summary>
    ///     Releases the compute buffers.
    /// </summary>
    private void ReleaseBuffers()
    {
        _hasBeenRelease = true;
        _vectorAddCuda.DestroyActionsAdd();
        _buffer1CS.Release();
        _buffer2CS.Release();
        _resultBufferCS.Release();

        _buffer1CUDA.Release();
        _buffer2CUDA.Release();
        _resultBufferCUDA.Release();
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
        for (int i = 0; i < _resultArrayCS.Length; i++)
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

using UnityEngine;

/// <summary>
///     This class performs vector addition using both GPU (via compute shader) and CPU,
///     then compares the results.
/// </summary>
public class VectorAddCS : MonoBehaviour
{
    // Serialized fields visible in the Unity Inspector
    [SerializeField] private int _arrayLength = 1000; // Length of the arrays to be added
    [SerializeField] private ComputeShader _computeShader; // Reference to the compute shader asset

    // Constants for buffer and kernel names used in the compute shader
    private readonly string _buffer1Name = "Array1";
    private readonly string _buffer2Name = "Array2";
    private readonly string _bufferResultName = "Result";
    private readonly string _kernelName = "VectorAdd";

    // Arrays to hold the random input data and result
    private float[] _array1;
    private float[] _array2;

    // ComputeBuffers for GPU calculations
    private ComputeBuffer _buffer1;
    private ComputeBuffer _buffer2;

    // Tracking variables for compute shader
    private int _currentArrayLength;
    private int _kernelHandle;
    private float[] _resultArray;
    private ComputeBuffer _resultBuffer;

    /// <summary>
    ///     Initializes the compute buffers when the script starts.
    /// </summary>
    private void Start()
    {
        InitializeBuffers();
    }

    /// <summary>
    ///     Performs vector addition each frame and compares GPU and CPU results.
    /// </summary>
    private void Update()
    {
        // Reinitialize buffers if array length has changed
        if (_currentArrayLength != _arrayLength)
        {
            ReleaseBuffers();
            InitializeBuffers();
        }

        // Generate random input arrays
        _array1 = GenerateRandomArray(_arrayLength);
        _array2 = GenerateRandomArray(_arrayLength);
        _resultArray = new float[_arrayLength];

        // Perform calculations on GPU and CPU
        float[] gpuSum = ComputeGPUSum();
        float[] cpuSum = ComputeCPUSum();

        // Compare results
        bool isEqual = CompareResults(gpuSum, cpuSum);
        if (isEqual == false)
        {
            Debug.LogError("GPU and CPU results do not match!");
        }
    }

    /// <summary>
    ///     Releases compute buffers when the script is disabled or destroyed.
    /// </summary>
    private void OnDisable()
    {
        ReleaseBuffers();
    }

    /// <summary>
    ///     Initializes compute buffers and sets up the compute shader.
    /// </summary>
    private void InitializeBuffers()
    {
        _currentArrayLength = _arrayLength;

        // Create compute buffers
        _buffer1 = new ComputeBuffer(_arrayLength, sizeof(float));
        _buffer2 = new ComputeBuffer(_arrayLength, sizeof(float));
        _resultBuffer = new ComputeBuffer(_arrayLength, sizeof(float));

        // Set up compute shader
        _kernelHandle = _computeShader.FindKernel(_kernelName);
        _computeShader.SetBuffer(_kernelHandle, _buffer1Name, _buffer1);
        _computeShader.SetBuffer(_kernelHandle, _buffer2Name, _buffer2);
        _computeShader.SetBuffer(_kernelHandle, _bufferResultName, _resultBuffer);
    }

    /// <summary>
    ///     Releases the compute buffers.
    /// </summary>
    private void ReleaseBuffers()
    {
        if (_buffer1 != null)
        {
            _buffer1.Release();
            _buffer2.Release();
            _resultBuffer.Release();
        }
    }

    /// <summary>
    ///     Generates an array of random float values.
    /// </summary>
    /// <param name="length">The length of the array to generate.</param>
    /// <returns>An array of random float values.</returns>
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
    ///     Performs vector addition using the GPU via compute shader.
    /// </summary>
    /// <returns>The resulting array after GPU computation.</returns>
    private float[] ComputeGPUSum()
    {
        // Set input data
        _buffer1.SetData(_array1);
        _buffer2.SetData(_array2);

        // Run the compute shader
        _computeShader.Dispatch(_kernelHandle, Mathf.CeilToInt(_arrayLength / 64f), 1, 1);

        // Retrieve results
        _resultBuffer.GetData(_resultArray);

        return _resultArray;
    }

    /// <summary>
    ///     Performs vector addition using the CPU.
    /// </summary>
    /// <returns>The resulting array after CPU computation.</returns>
    private float[] ComputeCPUSum()
    {
        float[] result = new float[_arrayLength];
        for (int i = 0; i < _arrayLength; i++)
        {
            result[i] = _array1[i] + _array2[i];
        }

        return result;
    }

    /// <summary>
    ///     Compares the results from GPU and CPU calculations.
    /// </summary>
    /// <param name="gpuResult">The result array from GPU computation.</param>
    /// <param name="cpuResult">The result array from CPU computation.</param>
    /// <returns>True if results match, false otherwise.</returns>
    private bool CompareResults(float[] gpuResult, float[] cpuResult)
    {
        bool isEqual = true;
        for (int i = 0; i < _arrayLength; i++)
        {
            if (Mathf.Abs(gpuResult[i] - cpuResult[i]) > Mathf.Epsilon)
            {
                isEqual = false;
                Debug.LogError($"Mismatch at index {i}: GPU = {gpuResult[i]}, CPU = {cpuResult[i]}");
            }
        }

        return isEqual;
    }
}

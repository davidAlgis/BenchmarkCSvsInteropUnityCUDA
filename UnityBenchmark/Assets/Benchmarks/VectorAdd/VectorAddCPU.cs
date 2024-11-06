using System.Diagnostics;

/// <summary>
/// A class for performing vector addition on the CPU and measuring the execution time.
/// </summary>
public class VectorAddCPU
{
    // Stopwatch for timing CPU execution
    private Stopwatch _stopwatch;

    // Arrays for vector addition
    private float[] _array1;
    private float[] _array2;
    private float[] _resultArray;

    /// <summary>
    /// Initializes the arrays with the specified size.
    /// </summary>
    /// <param name="arraySize">Size of the arrays.</param>
    public void Init(int arraySize)
    {
        _array1 = new float[arraySize];
        _array2 = new float[arraySize];
        _resultArray = new float[arraySize];
    }

    /// <summary>
    /// Computes the element-wise sum of two arrays and measures the execution time.
    /// </summary>
    /// <param name="result">The resulting array after addition.</param>
    /// <returns>The elapsed time in milliseconds to complete the vector addition.</returns>
    public float ComputeSum(out float[] result)
    {
        // Warm-up computation to ensure timing is correctly computed
        int warmStep = 1;
        for (int _ = 0; _ < warmStep; _++)
        {
            Computation();
        }

        // Start the stopwatch to time the vector addition
        _stopwatch = Stopwatch.StartNew();

        // Perform the addition and store the result
        result = Computation();

        // Stop the stopwatch and return the elapsed time
        _stopwatch.Stop();
        return (float)_stopwatch.Elapsed.TotalMilliseconds;
    }

    /// <summary>
    /// Performs element-wise addition of _array1 and _array2.
    /// </summary>
    /// <returns>The result of the addition.</returns>
    private float[] Computation()
    {
        for (int i = 0; i < _array1.Length; i++)
        {
            _resultArray[i] = _array1[i] + _array2[i]; // Add elements at each index
        }

        return _resultArray;
    }
}

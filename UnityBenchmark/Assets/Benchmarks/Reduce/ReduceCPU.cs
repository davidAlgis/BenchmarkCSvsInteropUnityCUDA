using System.Diagnostics;

/// <summary>
/// A reduce on mono thread CPU is equivalent to a simple sum operation
/// </summary>
public class ReduceCPU
{
    // Stopwatch for timing CPU execution
    private Stopwatch _stopwatch;

    // Array to hold values for summation; arrays are used for faster access than lists
    // Reference: https://www.jacksondunstan.com/articles/3058
    private float[] _arrayToSum;

    /// <summary>
    /// Initializes the array with the specified size.
    /// </summary>
    /// <param name="arraySize">Size of the array to initialize.</param>
    public void Init(int arraySize)
    {
        _arrayToSum = new float[arraySize];
    }

    /// <summary>
    /// Computes the sum of all elements in the array, with performance timing.
    /// </summary>
    /// <param name="result">The result of the summation.</param>
    /// <returns>The elapsed time in milliseconds to complete the summation.</returns>
    public float ComputeSum(out float result)
    {
        // Warm-up computation to reduce initial setup time effects on the timing
        int warmStep = 1;
        for (int _ = 0; _ < warmStep; _++)
        {
            Computation();
        }

        // Start the stopwatch to time the summation
        _stopwatch = Stopwatch.StartNew();

        // Perform the summation and store the result
        result = Computation();

        // Stop the stopwatch and return the elapsed time
        _stopwatch.Stop();
        return (float)_stopwatch.Elapsed.TotalMilliseconds;
    }

    /// <summary>
    /// Performs the summation of the elements in the array.
    /// </summary>
    /// <returns>The sum of the elements in the array.</returns>
    private float Computation()
    {
        float sum = 0;
        foreach (float f in _arrayToSum)
        {
            sum += f; // Accumulate each element into the sum
        }

        return sum;
    }
}

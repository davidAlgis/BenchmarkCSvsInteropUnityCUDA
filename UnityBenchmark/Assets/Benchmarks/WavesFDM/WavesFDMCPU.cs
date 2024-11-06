using System.Diagnostics;

/// <summary>
///     This class performs a resolution of the wave equation using the Finite Difference Method (FDM) on the CPU
///     and measures the execution time.
/// </summary>
public class WavesFDMCPU
{
    // Constants used in the wave equation resolution
    private float _a; // A = c^2 * dt^2 / dx^2
    private float _b; // B = 4 - 2 * A

    // Dimensions of the simulation
    private int _sizeTexture; // Size of the texture (assuming square grid)
    private int _depth; // Depth of the texture array (number of layers)

    // Height arrays representing H(t+dt), H(t), and H(t-dt)
    private float[,,] _htNew; // H(t+dt)
    private float[,,] _ht; // H(t)
    private float[,,] _htOld; // H(t-dt)

    // Stopwatch for timing CPU execution
    private Stopwatch _stopwatch;

    /// <summary>
    ///     Initializes the CPU-based wave simulation with the specified parameters.
    /// </summary>
    /// <param name="a">Constant A = c^2 * dt^2 / dx^2 used in the wave equation.</param>
    /// <param name="b">Constant B = 4 - 2 * A used in the wave equation.</param>
    /// <param name="sizeTexture">The size of the texture (grid) for simulation.</param>
    /// <param name="depth">The depth of the texture array (number of layers).</param>
    public void Init(float a, float b, int sizeTexture, int depth)
    {
        _a = a;
        _b = b;
        _sizeTexture = sizeTexture;
        _depth = depth;

        // Initialize the height arrays with zero values
        _htNew = new float[depth, sizeTexture, sizeTexture];
        _ht = new float[depth, sizeTexture, sizeTexture];
        _htOld = new float[depth, sizeTexture, sizeTexture];

        // Initialize the stopwatch
        _stopwatch = new Stopwatch();
    }

    /// <summary>
    ///     Sets the initial height values for H(t) and H(t-dt).
    /// </summary>
    /// <param name="ht">Height array at current time H(t).</param>
    /// <param name="htOld">Height array at previous time H(t-dt).</param>
    public void SetInitialHeights(float[,,] ht, float[,,] htOld)
    {
        if (ht.GetLength(0) != _depth || ht.GetLength(1) != _sizeTexture || ht.GetLength(2) != _sizeTexture)
        {
            throw new System.ArgumentException("Input ht array dimensions do not match the initialized size.");
        }

        if (htOld.GetLength(0) != _depth || htOld.GetLength(1) != _sizeTexture || htOld.GetLength(2) != _sizeTexture)
        {
            throw new System.ArgumentException("Input htOld array dimensions do not match the initialized size.");
        }

        // Copy input data to the internal arrays
        for (int k = 0; k < _depth; k++)
        {
            for (int i = 0; i < _sizeTexture; i++)
            {
                for (int j = 0; j < _sizeTexture; j++)
                {
                    _ht[k, i, j] = ht[k, i, j];
                    _htOld[k, i, j] = htOld[k, i, j];
                }
            }
        }
    }

    /// <summary>
    ///     Performs one iteration of the Finite Difference Method (FDM) to update the wave heights.
    ///     Measures and returns the execution time.
    /// </summary>
    /// <returns>The elapsed time in milliseconds for the computation.</returns>
    public float Update()
    {
        // Perform warm-up computations to stabilize timing
        int warmStep = 5;
        for (int i = 0; i < warmStep; i++)
        {
            Computation();
        }

        // Start the stopwatch to measure execution time
        _stopwatch.Restart();

        // Perform the computation
        Computation();

        // Stop the stopwatch and return the elapsed time
        _stopwatch.Stop();
        return (float)_stopwatch.Elapsed.TotalMilliseconds;
    }

    /// <summary>
    ///     Performs the Finite Difference Method (FDM) computation to update the wave heights.
    /// </summary>
    private void Computation()
    {
        for (int k = 0; k < _depth; k++)
        {
            for (int i = 1; i < _sizeTexture - 1; i++)
            {
                for (int j = 1; j < _sizeTexture - 1; j++)
                {
                    // Apply the FDM formula to compute H(t+dt)
                    _htNew[k, i, j] = _a * (
                        _ht[k, i + 1, j] + // H(t, i+1, j)
                        _ht[k, i - 1, j] + // H(t, i-1, j)
                        _ht[k, i, j + 1] + // H(t, i, j+1)
                        _ht[k, i, j - 1] // H(t, i, j-1)
                    ) + _b * _ht[k, i, j] - _htOld[k, i, j];
                }
            }
        }

        // Swap the height arrays for the next iteration
        SwapHeightArrays();
    }

    /// <summary>
    ///     Swaps the height arrays to prepare for the next iteration.
    /// </summary>
    private void SwapHeightArrays()
    {
        // Swap references: H(t-dt) = H(t), H(t) = H(t+dt), H(t+dt) is ready for the next update
        float[,,] temp = _htOld;
        _htOld = _ht;
        _ht = _htNew;
        _htNew = temp;
    }

    /// <summary>
    ///     Retrieves the current height array H(t).
    /// </summary>
    /// <returns>The current height array.</returns>
    public float[,,] GetCurrentHeight() => _ht;

    /// <summary>
    ///     Retrieves the new height array H(t+dt).
    /// </summary>
    /// <returns>The new height array.</returns>
    public float[,,] GetNewHeight() => _htNew;

    /// <summary>
    ///     Retrieves the old height array H(t-dt).
    /// </summary>
    /// <returns>The old height array.</returns>
    public float[,,] GetOldHeight() => _htOld;

    /// <summary>
    ///     Sets the current and old height arrays directly (useful for initializing with specific data).
    /// </summary>
    /// <param name="ht">Height array at current time H(t).</param>
    /// <param name="htOld">Height array at previous time H(t-dt).</param>
    public void SetHeights(float[,,] ht, float[,,] htOld)
    {
        SetInitialHeights(ht, htOld);
    }
}

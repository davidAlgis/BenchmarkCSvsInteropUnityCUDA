using System;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
///     This class handles the get data action using an external plugin for CUDA operations.
/// </summary>
public class ActionReduce : ActionUnity.ActionUnity
{
    // Define the name of the external plugin based on the build configuration
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private const string _dllbenchmarkPlugin = "d_BenchmarkPlugin"; // Debug build
#else
    private const string _dllbenchmarkPlugin = "BenchmarkPlugin"; // Release build
#endif

    [DllImport(_dllbenchmarkPlugin)]
    private static extern IntPtr createActionReduce(IntPtr arrayToSum, int arraySize);

    [DllImport(_dllbenchmarkPlugin)]
    private static extern float retrieveLastExecTimeCudaReduce(IntPtr action);

    /// <summary>
    ///     Constructor to initialize the ActionReduce object.
    /// </summary>
    /// <param name="arrayToSum">ComputeBuffer for the first input array.</param>
    /// <param name="arraySize">Size of the arrays.</param>
    public ActionReduce(ComputeBuffer arrayToSum, int arraySize) =>
        // Call the createActionReduce function from the plugin to create the action
        // and set the pointer to the created action object in _actionPtr
        _actionPtr = createActionReduce(arrayToSum.GetNativeBufferPtr(), arraySize);

    /// <summary>
    ///     Retrieves the last execution time of the CUDA operation.
    /// </summary>
    /// <returns>The last execution time in milliseconds.</returns>
    public float RetrieveLastExecTimeCuda() => retrieveLastExecTimeCudaReduce(_actionPtr);
}

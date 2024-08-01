using System;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
///     This class handles the get data action using an external plugin for CUDA operations.
/// </summary>
public class ActionGetData : ActionUnity.ActionUnity
{
    // Define the name of the external plugin based on the build configuration
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private const string _dllbenchmarkPlugin = "d_BenchmarkPlugin"; // Debug build
#else
    private const string _dllbenchmarkPlugin = "BenchmarkPlugin"; // Release build
#endif

    [DllImport(_dllbenchmarkPlugin)]
    private static extern IntPtr createActionGetData(IntPtr arrayToGet, int arraySize);

    [DllImport(_dllbenchmarkPlugin)]
    private static extern float retrieveLastExecTimeCudaGetData(IntPtr action);

    /// <summary>
    ///     Constructor to initialize the ActionGetData object.
    /// </summary>
    /// <param name="arrayToGet">ComputeBuffer for the first input array.</param>
    /// <param name="arraySize">Size of the arrays.</param>
    public ActionGetData(ComputeBuffer arrayToGet, int arraySize) =>
        // Call the createActionVectorAdd function from the plugin to create the action
        // and set the pointer to the created action object in _actionPtr
        _actionPtr = createActionGetData(arrayToGet.GetNativeBufferPtr(), arraySize);

    /// <summary>
    ///     Retrieves the last execution time of the CUDA operation.
    /// </summary>
    /// <returns>The last execution time in milliseconds.</returns>
    public float RetrieveLastExecTimeCuda() => retrieveLastExecTimeCudaGetData(_actionPtr);
}

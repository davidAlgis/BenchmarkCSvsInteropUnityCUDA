using System;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
///     This class handles the vector addition action using an external plugin for CUDA operations.
/// </summary>
public class ActionVectorAdd : ActionUnity.ActionUnity
{
    // Define the name of the external plugin based on the build configuration
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private const string _dllbenchmarkPlugin = "d_BenchmarkPlugin"; // Debug build
#else
    private const string _dllbenchmarkPlugin = "BenchmarkPlugin"; // Release build
#endif

    // Import the createActionVectorAdd function from the external plugin
    [DllImport(_dllbenchmarkPlugin)]
    private static extern IntPtr createActionVectorAdd(IntPtr array1, IntPtr array2, IntPtr arrayResult, int arraySize,
        int nbrElementToRetrieve);

    // Import the retrieveLastExecTimeCuda function from the external plugin
    [DllImport(_dllbenchmarkPlugin)]
    private static extern float retrieveLastExecTimeCudaVecAdd(IntPtr action);

    /// <summary>
    ///     Constructor to initialize the ActionVectorAdd object.
    /// </summary>
    /// <param name="array1">ComputeBuffer for the first input array.</param>
    /// <param name="array2">ComputeBuffer for the second input array.</param>
    /// <param name="arrayResult">ComputeBuffer for the result array.</param>
    /// <param name="arraySize">Size of the arrays.</param>
    /// <param name="nbrElementToRetrieve">
    ///     Defined the number of element that needs to be retrieve by CPU from the result
    ///     compute buffer.
    /// </param>
    public ActionVectorAdd(ComputeBuffer array1, ComputeBuffer array2, ComputeBuffer arrayResult, int arraySize,
        int nbrElementToRetrieve) =>
        // Call the createActionVectorAdd function from the plugin to create the action
        // and set the pointer to the created action object in _actionPtr
        _actionPtr = createActionVectorAdd(array1.GetNativeBufferPtr(), array2.GetNativeBufferPtr(),
            arrayResult.GetNativeBufferPtr(), arraySize, nbrElementToRetrieve);

    /// <summary>
    ///     Retrieves the last execution time of the CUDA operation.
    /// </summary>
    /// <returns>The last execution time in milliseconds.</returns>
    public float RetrieveLastExecTimeCuda() => retrieveLastExecTimeCudaVecAdd(_actionPtr);
}

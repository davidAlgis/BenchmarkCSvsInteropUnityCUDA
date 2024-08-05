using System;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
///     This class handles the vector addition action using an external plugin for CUDA operations.
/// </summary>
public class ActionWavesFDM : ActionUnity.ActionUnity
{
    // Define the name of the external plugin based on the build configuration
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private const string _dllbenchmarkPlugin = "d_BenchmarkPlugin"; // Debug build
#else
    private const string _dllbenchmarkPlugin = "BenchmarkPlugin"; // Release build
#endif

    [DllImport(_dllbenchmarkPlugin)]
    private static extern IntPtr createActionFDMWaves(IntPtr htNewPtr, IntPtr htPtr, IntPtr htOldPtr, int width,
        int height, int depth, float a, float b);

    [DllImport(_dllbenchmarkPlugin)]
    private static extern float retrieveLastExecTimeCudaFDMWaves(IntPtr action);

    public ActionWavesFDM(Texture2DArray htNewPtr, Texture2DArray htPtr, Texture2DArray htOldPtr, int width,
        int height, int depth, float a, float b) =>
        _actionPtr = createActionFDMWaves(htNewPtr.GetNativeTexturePtr(), htPtr.GetNativeTexturePtr(),
            htOldPtr.GetNativeTexturePtr(), width,
            height, depth, a, b);

    /// <summary>
    ///     Retrieves the last execution time of the CUDA operation.
    /// </summary>
    /// <returns>The last execution time in milliseconds.</returns>
    public float RetrieveLastExecTimeCuda() => retrieveLastExecTimeCudaFDMWaves(_actionPtr);
}

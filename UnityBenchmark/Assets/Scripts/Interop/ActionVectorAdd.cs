using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class ActionVectorAdd : ActionUnity.ActionUnity
{
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private const string _dllbenchmarkPlugin = "d_BenchmarkPlugin";
#else
        private const string _dllbenchmarkPlugin = "BenchmarkPlugin";
#endif

    [DllImport(_dllbenchmarkPlugin)]
    private static extern IntPtr createActionVectorAdd(IntPtr array1, IntPtr array2, IntPtr arrayResult, int arraySize);

    // we create the object MyAction in constructor of MyActionUnity
    public ActionVectorAdd(ComputeBuffer array1, ComputeBuffer array2, ComputeBuffer arrayResult, int arraySize) =>
        // the pointer toward our object MyAction is set in _actionPtr
        _actionPtr = createActionVectorAdd(array1.GetNativeBufferPtr(), array2.GetNativeBufferPtr(),
            arrayResult.GetNativeBufferPtr(), arraySize);
}

using ActionUnity;
using UnityEngine;

/// <summary>
///     This class solves waves equation with finite difference method using CUDA and handles the interop with Unity.
/// </summary>
public class WavesFDMCUDA : InteropHandler
{
    // Instance of the ActionWavesFDM class to handle waves equation resolution
    private ActionWavesFDM _actionWavesFDM;

    // Stores the current action name
    private string _currentActionName = "";

    // Index for the current array size
    private int _currentArraySizeIndex;

    public void InitializeActionsWavesFDM(Texture2DArray htNewPtr, Texture2DArray htPtr, Texture2DArray htOldPtr,
        int size,
        int depth, float a, float b)
    {
        // Create a unique action name based on the array size
        _currentActionName = "wavesFDM" + size;

        // Instantiate the ActionWavesFDM class with the provided buffers and array size
        _actionWavesFDM = new ActionWavesFDM(htNewPtr, htPtr, htOldPtr, size, size, depth, a, b);

        // Register the action with the interop handler
        RegisterActionUnity(_actionWavesFDM, _currentActionName);

        // Call the start function for the action
        CallFunctionStartInAction(_currentActionName);
    }

    /// <summary>
    ///     Solves waves equation with finite difference method using CUDA.
    /// </summary>
    /// <returns>
    ///     The last execution time of the CUDA operation in milliseconds. As the call of cuda is performs with render
    ///     thread (see readme.md of InteropUnityCUDA plugin) the execution time that is retrieve will be the one of the last
    ///     frame.
    ///     But in our case it's not a problem, as we only want to estimate the execution time.
    /// </returns>
    public float ComputeWavesFDM()
    {
        // Call the update function for the action
        CallFunctionUpdateInAction(_currentActionName);

        // Retrieve and return the last execution time from the CUDA operation
        float execTime = _actionWavesFDM.RetrieveLastExecTimeCuda();
        return execTime;
    }

    /// <summary>
    ///     Destroys the action.
    /// </summary>
    public void DestroyActionsWavesFDM()
    {
        // Call the destroy function for the action
        CallFunctionOnDestroyInAction(_currentActionName);
    }
}

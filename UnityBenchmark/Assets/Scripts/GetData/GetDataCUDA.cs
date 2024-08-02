using ActionUnity;
using UnityEngine;

/// <summary>
///     This class manages copy from GPU to CPU of a buffer using CUDA and handles the interop with Unity.
/// </summary>
public class GetDataCUDA : InteropHandler
{
    private ActionGetData _actionGetData;

    // Stores the current action name
    private string _currentActionName = "";

    // Index for the current array size
    private int _currentArraySizeIndex;

    /// <summary>
    ///     Initializes the get data action with CUDA.
    /// </summary>
    /// <param name="arraySize">The size of the arrays.</param>
    /// <param name="bufferToGet">ComputeBuffer to retrieve.</param>
    public void InitializeActionsGetData(int arraySize, ComputeBuffer bufferToGet)
    {
        // Create a unique action name based on the array size
        _currentActionName = "getData" + arraySize;

        // Instantiate the ActionVectorAdd class with the provided buffers and array size
        _actionGetData = new ActionGetData(bufferToGet, arraySize);

        // Register the action with the interop handler
        RegisterActionUnity(_actionGetData, _currentActionName);

        // Call the start function for the action
        CallFunctionStartInAction(_currentActionName);
    }

    /// <summary>
    ///     Copy the data from GPU to CPU using CUDA.
    /// </summary>
    /// <returns>
    ///     The last execution time of the CUDA operation in milliseconds. As the call of cuda is performs with render
    ///     thread (see readme.md of InteropUnityCUDA plugin) the execution time that is retrieve will be the one of the last
    ///     frame.
    ///     But in our case it's not a problem, as we only want to estimate the execution time.
    /// </returns>
    public float UpdateGetData()
    {
        // Call the update function for the action
        CallFunctionUpdateInAction(_currentActionName);

        // Retrieve and return the last execution time from the CUDA operation
        float execTime = _actionGetData.RetrieveLastExecTimeCuda();
        return execTime;
    }

    /// <summary>
    ///     Destroys the get data action.
    /// </summary>
    public void DestroyActionsGetData()
    {
        // Call the destroy function for the action
        CallFunctionOnDestroyInAction(_currentActionName);
    }
}

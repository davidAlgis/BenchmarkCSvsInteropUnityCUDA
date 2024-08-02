using ActionUnity;
using UnityEngine;

/// <summary>
///     This class apply a reduce on a buffer using CUDA and handles the interop with Unity.
/// </summary>
public class ReduceCUDA : InteropHandler
{
    private ActionReduce _actionReduce;

    // Stores the current action name
    private string _currentActionName = "";

    // Index for the current array size
    private int _currentArraySizeIndex;

    /// <summary>
    ///     Initializes the reduce action with CUDA.
    /// </summary>
    /// <param name="arraySize">The size of the arrays.</param>
    /// <param name="bufferToSum">ComputeBuffer to retrieve.</param>
    public void InitializeActionsReduce(int arraySize, ComputeBuffer bufferToSum)
    {
        // Create a unique action name based on the array size
        _currentActionName = "reduce" + arraySize;

        // Instantiate the ActionVectorAdd class with the provided buffers and array size
        _actionReduce = new ActionReduce(bufferToSum, arraySize);

        // Register the action with the interop handler
        RegisterActionUnity(_actionReduce, _currentActionName);

        // Call the start function for the action
        CallFunctionStartInAction(_currentActionName);
    }

    /// <summary>
    ///     Apply a reduce on a compute buffer using CUDA.
    /// </summary>
    /// <returns>
    ///     The last execution time of the CUDA operation in milliseconds. As the call of cuda is performs with render
    ///     thread (see readme.md of InteropUnityCUDA plugin) the execution time that is retrieve will be the one of the last
    ///     frame.
    ///     But in our case it's not a problem, as we only want to estimate the execution time.
    /// </returns>
    public float UpdateReduce()
    {
        // Call the update function for the action
        CallFunctionUpdateInAction(_currentActionName);

        // Retrieve and return the last execution time from the CUDA operation
        float execTime = _actionReduce.RetrieveLastExecTimeCuda();
        return execTime;
    }

    /// <summary>
    ///     Destroys the get data action.
    /// </summary>
    public void DestroyActionsReduce()
    {
        // Call the destroy function for the action
        CallFunctionOnDestroyInAction(_currentActionName);
    }
}

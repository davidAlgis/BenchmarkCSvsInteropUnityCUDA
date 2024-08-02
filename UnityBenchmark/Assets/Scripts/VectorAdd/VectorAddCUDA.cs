using ActionUnity;
using UnityEngine;

/// <summary>
///     This class manages vector addition using CUDA and handles the interop with Unity.
/// </summary>
public class VectorAddCUDA : InteropHandler
{
    // Instance of the ActionVectorAdd class to handle vector addition
    private ActionVectorAdd _actionVectorAdd;

    // Stores the current action name
    private string _currentActionName = "";

    // Index for the current array size
    private int _currentArraySizeIndex;

    /// <summary>
    ///     Initializes the vector addition action with CUDA.
    /// </summary>
    /// <param name="arraySize">The size of the arrays.</param>
    /// <param name="buffer1">ComputeBuffer for the first input array.</param>
    /// <param name="buffer2">ComputeBuffer for the second input array.</param>
    /// <param name="resultBuffer">ComputeBuffer for the result array.</param>
    /// <param name="nbrElementToRetrieve">
    ///     Defined the number of element that needs to be retrieve by CPU from the result
    ///     compute buffer.
    /// </param>
    public void InitializeActionsAdd(int arraySize, ComputeBuffer buffer1, ComputeBuffer buffer2,
        ComputeBuffer resultBuffer, int nbrElementToRetrieve)
    {
        // Create a unique action name based on the array size
        _currentActionName = "vectorAdd" + arraySize;

        // Instantiate the ActionVectorAdd class with the provided buffers and array size
        _actionVectorAdd = new ActionVectorAdd(buffer1, buffer2, resultBuffer, arraySize, nbrElementToRetrieve);

        // Register the action with the interop handler
        RegisterActionUnity(_actionVectorAdd, _currentActionName);

        // Call the start function for the action
        CallFunctionStartInAction(_currentActionName);
    }

    /// <summary>
    ///     Computes the sum of the arrays using CUDA.
    /// </summary>
    /// <returns>
    ///     The last execution time of the CUDA operation in milliseconds. As the call of cuda is performs with render
    ///     thread (see readme.md of InteropUnityCUDA plugin) the execution time that is retrieve will be the one of the last
    ///     frame.
    ///     But in our case it's not a problem, as we only want to estimate the execution time.
    /// </returns>
    public float ComputeSum()
    {
        // Call the update function for the action
        CallFunctionUpdateInAction(_currentActionName);

        // Retrieve and return the last execution time from the CUDA operation
        float execTime = _actionVectorAdd.RetrieveLastExecTimeCuda();
        return execTime;
    }

    /// <summary>
    ///     Destroys the vector addition action.
    /// </summary>
    public void DestroyActionsAdd()
    {
        // Call the destroy function for the action
        CallFunctionOnDestroyInAction(_currentActionName);
    }
}

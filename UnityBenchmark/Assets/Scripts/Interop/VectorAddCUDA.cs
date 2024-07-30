using ActionUnity;
using UnityEngine;

public class VectorAddCUDA : InteropHandler
{
    private ActionVectorAdd _actionVectorAdd;
    private string _currentActionName = "";
    private int _currentArraySizeIndex;

    public void InitializeActionsAdd(int arraySize, ComputeBuffer buffer1, ComputeBuffer buffer2,
        ComputeBuffer resultBuffer)
    {
        _currentActionName = "vectorAdd" + arraySize;
        _actionVectorAdd = new ActionVectorAdd(buffer1, buffer2, resultBuffer, arraySize);
        RegisterActionUnity(_actionVectorAdd, _currentActionName);
        CallFunctionStartInAction(_currentActionName);
    }

    public float ComputeSum()
    {
        CallFunctionUpdateInAction(_currentActionName);
        float execTime = _actionVectorAdd.RetrieveLastExecTimeCuda();
        return execTime;
    }

    public void DestroyActionsAdd()
    {
        CallFunctionOnDestroyInAction(_currentActionName);
    }
}

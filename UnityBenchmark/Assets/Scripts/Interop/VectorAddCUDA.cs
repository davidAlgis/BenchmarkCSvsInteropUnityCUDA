using System.Collections.Generic;
using ActionUnity;
using UnityEngine;

public class VectorAddCUDA : InteropHandler
{
    [SerializeField] private List<int> _arraySizes = new() { 1000, 10000, 100000 };
    private ComputeBuffer _buffer1;
    private ComputeBuffer _buffer2;

    private string _currentActionName = "";
    private int _currentArraySizeIndex;
    private ComputeBuffer _resultBuffer;

    protected override void InitializeActions()
    {
        _currentActionName = "vectorAdd" + _arraySizes[_currentArraySizeIndex];
        ActionVectorAdd myAction = new(_buffer1, _buffer2, _resultBuffer, _arraySizes[_currentArraySizeIndex]);
        RegisterActionUnity(myAction, _currentActionName);
        CallFunctionStartInAction(_currentActionName);
    }

    protected override void UpdateActions()
    {
        CallFunctionUpdateInAction(_currentActionName);
    }

    protected override void OnDestroyActions()
    {
        CallFunctionOnDestroyInAction(_currentActionName);
    }
}

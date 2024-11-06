using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

/// <summary>
///     This class manages the resolution of the wave equation using Compute Shaders, CUDA, and CPU implementations,
///     handles initialization, updates, and the display of the results.
/// </summary>
public class WavesFDMManager : BenchmarkManager
{
    [SerializeField] private ComputeShader _computeShaderFDMWaves;
    [SerializeField] private ComputeShader _computeShaderSwitchTex;

    [SerializeField] private float _dx = 0.1f;
    [SerializeField] private float _dt = 0.5f;
    [SerializeField] private RawImage _rawImageCS;
    [SerializeField] private RawImage _rawImageCUDA;

    [SerializeField] private int _volumeDepth = 10;
    [SerializeField] private WavesFDMCUDA _wavesFDMCUDA;
    [SerializeField] private bool _displayResult;

    // Variables for wave equation constants
    private float _a;
    private float _b;

    // Compute buffers and arrays
    private ComputeBuffer _bufferPixelCS;
    private float[] _pixelArray;

    // Flags and state variables
    private bool _hasBeenReleased = true;

    // RenderTextures for Compute Shader implementation
    private RenderTexture _ht;
    private RenderTexture _htNew;
    private RenderTexture _htOld;

    // Texture2DArray for CUDA implementation
    private Texture2DArray _htCUDA;
    private Texture2DArray _htNewCUDA;
    private Texture2DArray _htOldCUDA;

    // Textures for displaying results
    private Texture2D _textureForDisplayCS;
    private Texture2D _textureForDisplayCUDA;

    // Compute Shader implementation instance
    private WavesFDMCS _wavesFDMCS;

    // CPU implementation instance
    private WavesFDMCPU _wavesFDMCPU;

    // CPU height arrays
    private float[,,] _cpuHtNew;
    private float[,,] _cpuHt;
    private float[,,] _cpuHtOld;

    private void OnDestroy()
    {
        if (!_hasBeenReleased)
        {
            ReleaseGraphics();
        }
    }

    /// <summary>
    ///     Initializes the compute shader, CUDA handlers, CPU simulation, and sets up the initial state.
    /// </summary>
    protected override void Initialize()
    {
        base.Initialize();

        // Define constants to respect the CFL condition c^2 * dt^2 / dx^2 <= 0.5
        float c = Mathf.Sqrt(0.45f) * _dx / _dt;
        _a = c * c * _dt * _dt / (_dx * _dx);
        _b = 2 - 4 * _a;

        // Initialize CUDA handler
        _wavesFDMCUDA.InitializeInteropHandler();

        // Initialize Compute Shader implementation
        _wavesFDMCS = new WavesFDMCS();

        // Initialize CPU implementation
        _wavesFDMCPU = new WavesFDMCPU();

        // Initialize resources
        ReInitialize();
    }

    /// <summary>
    ///     Updates the profiling process before recording starts.
    /// </summary>
    protected override void UpdateBeforeRecord()
    {
        // Perform warm-up computations to stabilize timings
        int warmStep = 5;
        for (int i = 0; i < warmStep; i++)
        {
            // Compute Shader implementation
            _wavesFDMCS.Update(_ht.width, _ht.height, _ht.volumeDepth, _bufferPixelCS, ref _pixelArray);

            // CUDA implementation
            _wavesFDMCUDA.ComputeWavesFDM();

            // CPU implementation
            _wavesFDMCPU.Update();
        }

        base.UpdateBeforeRecord();
    }

    /// <summary>
    ///     Updates the execution time for Compute Shader, CUDA, and CPU implementations, and refreshes the display.
    /// </summary>
    /// <param name="gpuExecutionTimeCS">Output parameter for the compute shader execution time.</param>
    /// <param name="gpuExecutionTimeCUDA">Output parameter for the CUDA execution time.</param>
    /// <param name="cpuExecutionTime">Output parameter for the CPU execution time.</param>
    protected override void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA,
        out float cpuExecutionTime)
    {
        int arraySize = _arraySizes[_currentArraySizeIndex];
        _titleText.text = $"Waves FDM - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";

        // Compute Shader implementation
        gpuExecutionTimeCS =
            _wavesFDMCS.Update(_ht.width, _ht.height, _ht.volumeDepth, _bufferPixelCS, ref _pixelArray);

        // CUDA implementation
        gpuExecutionTimeCUDA = _wavesFDMCUDA.ComputeWavesFDM();

        // CPU implementation
        cpuExecutionTime = _wavesFDMCPU.Update();

        // Optionally update the display texture
        UpdateDisplayTexture();

        // Optionally compare results
        // CompareResults();
    }

    /// <summary>
    ///     Reinitializes resources for the next array size in the benchmark.
    /// </summary>
    protected override void ReInitialize()
    {
        base.ReInitialize();
        int size = _arraySizes[_currentArraySizeIndex];

        if (!_hasBeenReleased)
        {
            ReleaseGraphics();
        }

        // Initialize RenderTextures for Compute Shader implementation
        _htNew = CreateRenderTexture(size, size, _volumeDepth);
        _ht = CreateRenderTexture(size, size, _volumeDepth);
        _htOld = CreateRenderTexture(size, size, _volumeDepth);

        // Initialize Texture2DArray for CUDA implementation
        _htNewCUDA = CreateTexture2DArray(size, size, _volumeDepth);
        _htCUDA = CreateTexture2DArray(size, size, _volumeDepth);
        _htOldCUDA = CreateTexture2DArray(size, size, _volumeDepth);

        // Initialize Compute Buffer
        _bufferPixelCS = new ComputeBuffer(1, sizeof(float));
        _pixelArray = new[] { 0.0f };

        // Initialize CPU height arrays
        _cpuHtNew = new float[_volumeDepth, size, size];
        _cpuHt = new float[_volumeDepth, size, size];
        _cpuHtOld = new float[_volumeDepth, size, size];

        // Initialize textures with some values if needed
        InitializeTextures(_ht, _htNew, _htOld);
        InitializeCPUHeights(_cpuHt, _cpuHtNew, _cpuHtOld);

        // Copy RenderTextures to Texture2DArray for CUDA implementation
        CopyRenderTextureToTexture2DArray(_htNew, _htNewCUDA);
        CopyRenderTextureToTexture2DArray(_ht, _htCUDA);
        CopyRenderTextureToTexture2DArray(_htOld, _htOldCUDA);

        // Initialize the display textures
        _textureForDisplayCS = new Texture2D(size, size, TextureFormat.RFloat, false);
        _textureForDisplayCUDA = new Texture2D(size, size, TextureFormat.RFloat, false);

        if (_displayResult)
        {
            _rawImageCS.texture = _textureForDisplayCS;
            _rawImageCUDA.texture = _textureForDisplayCUDA;
        }
        else
        {
            _rawImageCS.gameObject.SetActive(false);
            _rawImageCUDA.gameObject.SetActive(false);
        }

        // Initialize Compute Shader implementation
        _wavesFDMCS.Init(ref _htNew, ref _ht, ref _htOld, _computeShaderFDMWaves, _computeShaderSwitchTex, _a, _b, size,
            _volumeDepth, _bufferPixelCS);

        // Initialize CUDA implementation
        _wavesFDMCUDA.InitializeActionsWavesFDM(_htNewCUDA, _htCUDA, _htOldCUDA, size, _volumeDepth, _a, _b);

        // Initialize CPU implementation
        _wavesFDMCPU.Init(_a, _b, size, _volumeDepth);
        _wavesFDMCPU.SetInitialHeights(_cpuHt, _cpuHtOld);

        // Perform initial updates
        _wavesFDMCS.Update(_ht.width, _ht.height, _ht.volumeDepth, _bufferPixelCS, ref _pixelArray);
        _wavesFDMCUDA.ComputeWavesFDM();
        _wavesFDMCPU.Update();

        // Ensure CUDA textures are synchronized with Compute Shader textures
        CopyRenderTextureToTexture2DArray(_htNew, _htNewCUDA);
        CopyRenderTextureToTexture2DArray(_ht, _htCUDA);
        CopyRenderTextureToTexture2DArray(_htOld, _htOldCUDA);

        _hasBeenReleased = false;
    }

    /// <summary>
    ///     Creates a RenderTexture with the specified dimensions and volume depth.
    /// </summary>
    private RenderTexture CreateRenderTexture(int width, int height, int depth)
    {
        RenderTexture texture = new(width, height, 0, RenderTextureFormat.RFloat)
        {
            enableRandomWrite = true,
            dimension = TextureDimension.Tex2DArray,
            volumeDepth = depth
        };
        texture.Create();
        return texture;
    }

    /// <summary>
    ///     Creates a Texture2DArray with the specified dimensions and volume depth.
    /// </summary>
    private Texture2DArray CreateTexture2DArray(int width, int height, int depth)
    {
        Texture2DArray textureArray = new(width, height, depth, TextureFormat.RFloat, false, true);
        textureArray.Apply();
        return textureArray;
    }

    /// <summary>
    ///     Initializes the specified RenderTextures with default values and a central circle.
    /// </summary>
    private void InitializeTextures(params RenderTexture[] textures)
    {
        foreach (RenderTexture texture in textures)
        {
            for (int layer = 0; layer < _volumeDepth; layer++)
            {
                RenderTexture.active = texture;
                Texture2D tex = new(texture.width, texture.height, TextureFormat.RFloat, false);

                // Initialize all pixels to 0
                var pixels = new Color[texture.width * texture.height];
                for (int i = 0; i < pixels.Length; i++)
                {
                    pixels[i] = new Color(0f, 0, 0, 0);
                }

                // Calculate the center and radius of the circle
                int centerX = texture.width / 2;
                int centerY = texture.height / 2;
                int radius = Mathf.Min(texture.width, texture.height) / 10;

                // Set the pixels inside the circle to 1
                for (int y = 0; y < texture.height; y++)
                {
                    for (int x = 0; x < texture.width; x++)
                    {
                        int dx = x - centerX;
                        int dy = y - centerY;
                        if (dx * dx + dy * dy <= radius * radius)
                        {
                            pixels[y * texture.width + x] = new Color(1f, 0, 0, 0);
                        }
                    }
                }

                tex.SetPixels(pixels);
                tex.Apply();

                Graphics.CopyTexture(tex, 0, 0, texture, layer, 0);

                // Clean up
                Destroy(tex);
            }
        }
    }

    /// <summary>
    ///     Initializes the CPU height arrays with default values and a central circle.
    /// </summary>
    private void InitializeCPUHeights(float[,,] ht, float[,,] htNew, float[,,] htOld)
    {
        int size = _arraySizes[_currentArraySizeIndex];
        for (int k = 0; k < _volumeDepth; k++)
        {
            // Initialize all values to 0
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    ht[k, i, j] = 0f;
                    htNew[k, i, j] = 0f;
                    htOld[k, i, j] = 0f;
                }
            }

            // Set a central circle to 1
            int centerX = size / 2;
            int centerY = size / 2;
            int radius = size / 10;

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    int dx = i - centerX;
                    int dy = j - centerY;
                    if (dx * dx + dy * dy <= radius * radius)
                    {
                        ht[k, i, j] = 1f;
                    }
                }
            }
        }
    }

    /// <summary>
    ///     Copies the contents of a RenderTexture to a Texture2DArray.
    /// </summary>
    private void CopyRenderTextureToTexture2DArray(RenderTexture renderTexture, Texture2DArray textureArray)
    {
        for (int layer = 0; layer < textureArray.depth; layer++)
        {
            Graphics.CopyTexture(renderTexture, layer, 0, textureArray, layer, 0);
        }
    }

    /// <summary>
    ///     Updates the display texture by copying data from the height textures.
    /// </summary>
    private void UpdateDisplayTexture()
    {
        if (_displayResult)
        {
            // Copy the first layer of the RenderTexture (_ht) to _textureForDisplayCS
            Graphics.CopyTexture(_ht, 0, 0, _textureForDisplayCS, 0, 0);

            // Copy the first layer of the Texture2DArray (_htCUDA) to _textureForDisplayCUDA
            Graphics.CopyTexture(_htCUDA, 0, 0, _textureForDisplayCUDA, 0, 0);
        }
    }

    /// <summary>
    ///     Releases the compute buffers and textures.
    /// </summary>
    private void ReleaseGraphics()
    {
        _hasBeenReleased = true;

        if (_ht != null)
        {
            _ht.Release();
            _ht = null;
        }

        if (_htNew != null)
        {
            _htNew.Release();
            _htNew = null;
        }

        if (_htOld != null)
        {
            _htOld.Release();
            _htOld = null;
        }

        _bufferPixelCS?.Release();
        _bufferPixelCS = null;
    }

    /// <summary>
    ///     Compares the results of CPU and GPU computations to verify correctness.
    /// </summary>
    private void CompareResults()
    {
        // Retrieve the current height from CPU simulation
        float[,,] cpuCurrentHt = _wavesFDMCPU.GetCurrentHeight();

        // Retrieve the current height from GPU simulation (Compute Shader)
        RenderTexture.active = _ht;
        Texture2D tex = new(_ht.width, _ht.height, TextureFormat.RFloat, false);
        tex.ReadPixels(new Rect(0, 0, _ht.width, _ht.height), 0, 0);
        tex.Apply();

        bool isEqual = true;
        float epsilon = 0.0001f; // Tolerance for floating-point comparison
        int size = _ht.width;

        for (int k = 0; k < _volumeDepth; k++)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    float cpuValue = cpuCurrentHt[k, i, j];
                    float gpuValue = tex.GetPixel(j, i).r; // Note: Texture2D uses (x, y)

                    if (Mathf.Abs(cpuValue - gpuValue) > epsilon)
                    {
                        isEqual = false;
                        Debug.LogError(
                            $"Mismatch at layer {k}, position ({i}, {j}): CPU = {cpuValue}, GPU = {gpuValue}");
                        break;
                    }
                }

                if (!isEqual)
                {
                    break;
                }
            }

            if (!isEqual)
            {
                break;
            }
        }

        if (isEqual)
        {
            Debug.Log("CPU and GPU results match.");
        }
        else
        {
            Debug.LogError("CPU and GPU results do not match.");
        }

        // Clean up
        RenderTexture.active = null;
        Destroy(tex);
    }
}

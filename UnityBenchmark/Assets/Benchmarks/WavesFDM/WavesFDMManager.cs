using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

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
    private float _a;
    private float _b;
    private ComputeBuffer _bufferPixelCS;

    private bool _hasBeenReleased = true;

    private RenderTexture _ht;

    private Texture2DArray _htCUDA;
    private RenderTexture _htNew;
    private Texture2DArray _htNewCUDA;
    private RenderTexture _htOld;
    private Texture2DArray _htOldCUDA;
    private float[] _pixelArray;
    private Texture2D _textureForDisplayCS;
    private Texture2D _textureForDisplayCUDA;

    private WavesFDMCS _wavesFDMCS;

    private void OnDestroy()
    {
        if (!_hasBeenReleased)
        {
            ReleaseGraphics();
        }
    }

    protected override void Initialize()
    {
        // define to respect the CFL c^2*dt^2/dx^2 <= 0.5
        float c = Mathf.Sqrt(0.45f) * _dx / _dt;
        _a = c * c * _dt * _dt / (_dx * _dx);
        _b = 2 - 4 * _a;
        base.Initialize();
        _wavesFDMCUDA.InitializeInteropHandler();
        _wavesFDMCS = new WavesFDMCS();
        ReInitialize();
    }

    protected override void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA)
    {
        int arraySize = _arraySizes[_currentArraySizeIndex];
        _titleText.text = $"Waves FDM - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";
        gpuExecutionTimeCS = 0.0f;
        gpuExecutionTimeCUDA = 0.0f;
        gpuExecutionTimeCS =
            _wavesFDMCS.Update(_ht.width, _ht.height, _ht.volumeDepth, _bufferPixelCS, ref _pixelArray);
        gpuExecutionTimeCUDA = _wavesFDMCUDA.ComputeWavesFDM();

        // Update the display texture
        UpdateDisplayTexture();
    }

    protected override void ReInitialize()
    {
        base.ReInitialize();
        int size = _arraySizes[_currentArraySizeIndex];
        if (_hasBeenReleased == false)
        {
            ReleaseGraphics();
        }

        // Allocate and initialize RenderTextures
        _htNew = CreateRenderTexture(size, size, _volumeDepth);
        _ht = CreateRenderTexture(size, size, _volumeDepth);
        _htOld = CreateRenderTexture(size, size, _volumeDepth);

        _htNewCUDA = CreateTexture2DArray(size, size, _volumeDepth);
        _htCUDA = CreateTexture2DArray(size, size, _volumeDepth);
        _htOldCUDA = CreateTexture2DArray(size, size, _volumeDepth);
        _bufferPixelCS = new ComputeBuffer(1, sizeof(float));
        _pixelArray = new[] { 0.0f };

        // Initialize textures with some values if needed
        InitializeTextures(_ht, _htNew, _htOld);

        // Copy RenderTextures to Texture2DArrays
        CopyRenderTextureToTexture2DArray(_htNew, _htNewCUDA);
        CopyRenderTextureToTexture2DArray(_ht, _htCUDA);
        CopyRenderTextureToTexture2DArray(_htOld, _htOldCUDA);

        // Initialize the display texture
        _textureForDisplayCS = new Texture2D(size, size, TextureFormat.RFloat, false);
        _textureForDisplayCUDA = new Texture2D(size, size, TextureFormat.RFloat, false);
        _rawImageCS.texture = _textureForDisplayCS;
        _rawImageCUDA.texture = _textureForDisplayCUDA;
        _wavesFDMCS.Init(ref _htNew, ref _ht,
            ref _htOld, _computeShaderFDMWaves, _computeShaderSwitchTex, _a, _b, _arraySizes[_currentArraySizeIndex],
            _volumeDepth, _bufferPixelCS);
        _wavesFDMCUDA.InitializeActionsWavesFDM(_htNewCUDA, _htCUDA, _htOldCUDA,
            size, _volumeDepth, _a, _b);
        _wavesFDMCUDA.ComputeWavesFDM();
        CopyRenderTextureToTexture2DArray(_htNew, _htNewCUDA);
        CopyRenderTextureToTexture2DArray(_ht, _htCUDA);
        CopyRenderTextureToTexture2DArray(_htOld, _htOldCUDA);
        _hasBeenReleased = false;
    }

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

    private Texture2DArray CreateTexture2DArray(int width, int height, int depth)
    {
        Texture2DArray textureArray = new(width, height, depth, TextureFormat.RFloat, false, true);
        textureArray.Apply();
        return textureArray;
    }

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
            }
        }
    }

    private void CopyRenderTextureToTexture2DArray(RenderTexture renderTexture, Texture2DArray textureArray)
    {
        for (int layer = 0; layer < textureArray.depth; layer++)
        {
            Graphics.CopyTexture(renderTexture, layer, 0, textureArray, layer, 0);
        }
    }

    private void UpdateDisplayTexture()
    {
        // Copy the first texture of the array (_ht) to _textureForDisplay
        Graphics.CopyTexture(_ht, 1, 0, _textureForDisplayCS, 0, 0);

        Graphics.CopyTexture(_htCUDA, 1, 0, _textureForDisplayCUDA, 0, 0);
    }

    /// <summary>
    ///     Releases the render textures.
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

        _bufferPixelCS.Release();
    }
}
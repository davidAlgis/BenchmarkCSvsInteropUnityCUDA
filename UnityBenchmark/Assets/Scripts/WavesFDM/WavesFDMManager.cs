using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

public class WavesFDMManager : BenchmarkManager
{
    [SerializeField] private ComputeShader _computeShader;

    [SerializeField] private float _dx = 0.1f;
    [SerializeField] private float _dt = 0.5f;
    [SerializeField] private RawImage _rawImageOneTexture;

    [SerializeField] private int _volumeDepth = 10;
    private float _a;
    private float _b;

    private bool _hasBeenReleased;

    private RenderTexture _ht;
    private RenderTexture _htNew;
    private RenderTexture _htOld;
    private Texture2D _textureForDisplay;

    private WavesFDMCS _wavesFDMCS;

    private void OnDestroy()
    {
        if (!_hasBeenReleased)
        {
            ReleaseTextures();
        }
    }

    protected override void Initialize()
    {
        // define to respect the CFL c^2*dt^2/dx^2 <= 0.5
        float c = Mathf.Sqrt(0.45f) * _dx / _dt;
        _a = c * c * _dt * _dt / (_dx * _dx);
        _b = 2 - 4 * _a;
        base.Initialize();
        _wavesFDMCS = new WavesFDMCS();
        ReInitialize();
    }

    protected override void UpdateMainRecord(out float gpuExecutionTimeCS, out float gpuExecutionTimeCUDA)
    {
        int arraySize = _arraySizes[_currentArraySizeIndex];
        _titleText.text = $"Waves FDM - {arraySize} - Sample {_currentSampleCount}/{_numSamplesPerSize}";
        gpuExecutionTimeCS =
            _wavesFDMCS.Update(ref _htNew, ref _ht, ref _htOld, _ht.width, _ht.height, _ht.volumeDepth);
        gpuExecutionTimeCUDA = 0.0f; // Assuming CUDA is not used in this case

        // Update the display texture
        UpdateDisplayTexture();
    }

    protected override void ReInitialize()
    {
        base.ReInitialize();
        print("reinit");
        int size = _arraySizes[_currentArraySizeIndex];

        // Allocate and initialize RenderTextures
        _htNew = CreateRenderTexture(size, size, _volumeDepth);
        _ht = CreateRenderTexture(size, size, _volumeDepth);
        _htOld = CreateRenderTexture(size, size, _volumeDepth);

        // Initialize textures with some values if needed
        InitializeTextures(_ht, _htNew, _htOld);

        // Initialize the display texture
        _textureForDisplay = new Texture2D(size, size, TextureFormat.RFloat, false);
        _rawImageOneTexture.texture = _textureForDisplay;
        _wavesFDMCS.Init(_computeShader, _a, _b, _arraySizes[_currentArraySizeIndex]);
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

    private void InitializeTextures(params RenderTexture[] textures)
    {
        foreach (RenderTexture texture in textures)
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
            Graphics.Blit(tex, texture);
        }
    }

    private void UpdateDisplayTexture()
    {
        // Copy the first texture of the array (_ht) to _textureForDisplay
        RenderTexture.active = _ht;
        _textureForDisplay.ReadPixels(new Rect(0, 0, _ht.width, _ht.height), 0, 0);
        _textureForDisplay.Apply();
        RenderTexture.active = null;
    }

    /// <summary>
    ///     Releases the render textures.
    /// </summary>
    private void ReleaseTextures()
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
    }
}

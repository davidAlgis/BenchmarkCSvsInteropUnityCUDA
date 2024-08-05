using System.Linq;
using UnityEditor;
using UnityEngine;

public class BuildScript
{
    [MenuItem("Build/BuildGL")]
    public static void BuildGL()
    {
        string[] scenes = GetScenesFromBuildSettings();
        BuildPipeline.BuildPlayer(scenes, "build/GLBuild/UnityBenchmark.exe", BuildTarget.StandaloneWindows64,
            BuildOptions.None);
        Debug.Log("BuildGL completed successfully.");
    }

    [MenuItem("Build/BuildDX11")]
    public static void BuildDX11()
    {
        string[] scenes = GetScenesFromBuildSettings();
        BuildPipeline.BuildPlayer(scenes, "build/DX11Build/UnityBenchmark.exe", BuildTarget.StandaloneWindows64,
            BuildOptions.None);
        Debug.Log("BuildDX11 completed successfully.");
    }

    private static string[] GetScenesFromBuildSettings()
    {
        return EditorBuildSettings.scenes.Where(scene => scene.enabled).Select(scene => scene.path).ToArray();
    }
}

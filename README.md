# BenchmarkCSvsInteropUnityCUDA (Work in progress)

Benchmark between Compute Shader of Unity against [InteropUnityCUDA](https://github.com/davidAlgis/InteropUnityCUDA). 

__This repository is currently under development !!!__

## Get started

This section explain the procedure to make the unity project works : 

1. Install Unity 2022.3.8 with Unity Hub. While it's downloading/installing...
2. [Package the InteropUnityCUDA using the dedicated documentation](PluginBenchmark\Vendors\InteropUnityCUDA\Plugin\Documentation\GenerateUnityPackage.md), the package should be place at the root of the folder `InteropUnityCUDA` in `PluginBenchmark\Vendors`. 
3. Generate and compile the project in `PluginBenchmark folder` :
```
cmake -B build
cmake -build build --config Debug
cmake -build build --config Release
```
4. Launch the Unity project of `UnityBenchmark`.

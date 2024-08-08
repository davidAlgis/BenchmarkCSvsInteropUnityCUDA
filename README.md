# BenchmarkCSvsInteropUnityCUDA

This repository propose four tests to benchmark the performances between Compute Shader of Unity against a tools of interoperability between Unity and CUDA: [InteropUnityCUDA](https://github.com/davidAlgis/InteropUnityCUDA).

## Tests Overview

The tests included in this benchmark are:

1. GetData: This test measures the time it takes to copy a compute buffer of size NN from GPU to CPU. In Unity, this is done using GetData, and in InteropUnityCUDA, it is done using cudaMemcpy.
2. VectorAdd: This test sums two vectors of floats of size NN, stores the result in another vector, and then copies the first float of the resulting vector from GPU to CPU.
3. Reduce: This test applies a parallel reduction on an array of size NN and copies the result from GPU to CPU. The CUDA implementation uses the optimized cub library, while the compute shader implementation is custom-built.
4. WavesFDM: This test solves the 2D wave equation using the finite difference method. It involves evolving the height of a 2D domain over time using a stencil-based explicit scheme and then copying the result from GPU to CPU.

## Results

The tests were executed on an Nvidia GeForce RTX 2070 Super and an Intel Core i7-10700, with each test run 1000 times for various problem sizes $N$ to ensure reliable results. The performance comparisons revealed the following:

1. GetData Test: OpenGL and CUDA performed equivalently, with both outperforming DirectX 11 for $N>10^6$.
2. VectorAdd Test: CUDA and OpenGL showed high variance in performance, while DirectX 11 maintained stable performance. For larger arrays $N>10^5$, CUDA and DirectX 11 outperformed OpenGL.
3. Reduce Test: CUDA demonstrated a significant performance advantage for $N>10^5$, thanks to the optimized cub library. DirectX 11 and OpenGL were comparable for smaller arrays but lagged behind CUDA for larger ones.
4. WavesFDM Test: CUDA consistently outperformed both graphics APIs for all $N$.

Overall, CUDA provides superior performance, especially for larger data sets and complex operations, making InteropUnityCUDA a valuable tool for leveraging CUDA's capabilities within Unity for performance-critical applications.

## Get started

This section explain the procedure to make the unity project works : 

### Manual Installation

1. Install Unity 2022.3.8 (it might work on any version of Unity greater than 2021, but I didn't test it) with Unity Hub. While it's downloading/installing...
2. [Package the InteropUnityCUDA using the dedicated documentation](PluginBenchmark/Vendors/InteropUnityCUDA/Plugin/Documentation/GenerateUnityPackage.md), the package should be place at the root of the folder `InteropUnityCUDA` in `PluginBenchmark\Vendors`. 
3. Generate and compile the project in `PluginBenchmark folder` :
```sh
cmake -B build
cmake -build build --config Debug
cmake -build build --config Release
```
4. Launch the Unity project of `UnityBenchmark`.

### Automatic Installation

1. Install Unity 2022.3.8 (it might work on any version of Unity greater than 2021, but I didn't test it) with Unity Hub.
2. Call the script `benchmark.py` in the root folder (see section below). 

## Benchmark automatic script 

This Python script is designed to automate the process of benchmarking.

### Installation

Before running the script, you need to install the required Python libraries. You can install them using the provided requirements.txt file.

```
pip install -r requirements.txt
```

### Usage

Ensure you have Python 3.6 or newer installed on your system. Clone this repository or download the script. To use the script, run it from the command line with the desired options:

```sh
python main.py [options]
```
### Options

- ``-c``, ``--cuda-path <path>``: Path to the CUDA toolkit (default: ``C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v12.3``).
- ``-u``, ``--unityBin <path>``: Path to the Unity executable (default: ``C://Program Files//Unity//Hub//Editor//2022.3.8f1//Editor//Unity.exe``).
- ``-p``, ``--plugin <boolean>``: If true, will configure and build the plugin benchmark (default: true).
- ``-pa``, ``--package-interop <boolean>``: If true, will package InteropUnityCUDA (default: true).
- ``-bu``, ``--build-unity <boolean>``: If true, will build the Unity benchmark project (default: true).
- ``-b``, ``--benchmark <boolean>``: If true, will benchmark by launching the Unity project (default: true).
- ``-d``, ``--draw <boolean>``: If true, will draw the graph of comparison (default: true).
- ``-ba``, ``--batchmode <boolean>``: If true, will display the benchmark in batch mode (default: false).
- ``-n``, ``--number-tests <int>``: Number of times the benchmark should be launched (default: 1).
- ``-ns``, ``--number-samples <int>``: Number of samples per size to update the Config.json (default: 1000).
- ``-h``, ``--help``: Display help information showing all command-line options.

### Example

To configure and build the plugin benchmark, package InteropUnityCUDA, build the Unity benchmark project, run the benchmarks, and generate comparison graphs, you would use:

```sh
python main.py -c "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v12.3" -u "C://Program Files//Unity//Hub//Editor//2022.3.8f1//Editor//Unity.exe" -p true -pa true -bu true -b true -d true -ba false -n 1 -ns 1000
```

This will perform all the specified actions, including building and benchmarking the Unity project and generating comparison graphs.

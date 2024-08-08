import os
import subprocess
import argparse
import sys
import shutil
import json


def run_command(*args, **kwargs):
    process = subprocess.Popen(
        *args, **kwargs, stdout=sys.stdout, stderr=sys.stderr)
    stdout, stderr = process.communicate()
    return process.returncode


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def buildConfig(script_dir, project_dir, cuda_path):
    cmake_args = ['cmake', "-B", "build",
                  f"-DCMAKE_GENERATOR_TOOLSET=cuda={cuda_path}", "-D CMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    print_green("Execute cmake with args: ", cmake_args)
    run_command(cmake_args, cwd=project_dir)


def compileProjects(project_dir, buildType):
    cmake_args = ['cmake', "--build", "build"]
    cmake_args.append("--config")
    cmake_args.append(f"{buildType}")
    print_green("Execute cmake with args: ", cmake_args)
    ret = run_command(cmake_args, cwd=project_dir)
    return ret


def print_green(*args, **kwargs):
    print("\033[92m", *args, "\033[0m", **kwargs)


def print_red(*args, **kwargs):
    print("\033[91m", *args, "\033[0m", **kwargs)


def create_config_file(build_dir, num_samples):
    config_path = os.path.join(build_dir, 'Config.json')
    config = {
        'NumSamplesPerSize': num_samples
    }
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)
    print_green(f"Config file created: {config_path}")


def package_interop(script_dir):
    print_green("Package InteropUnityCUDA...")
    interop_unity_cuda_dir = os.path.join(
        script_dir, 'PluginBenchmark', 'Vendors', 'InteropUnityCUDA')
    package_script_path = os.path.join(
        interop_unity_cuda_dir, 'Plugin', 'buildtools', 'packageUnity.py')
    if os.path.exists(interop_unity_cuda_dir):
        packaged_args = ['python', package_script_path]
        print_green("Execute package script: ", packaged_args)
        run_command(packaged_args, cwd=script_dir)


def generate_compile_plugin(script_dir, cuda_path):
    print_green("Configure Benchmark Plugin...")
    project_dir = os.path.join(script_dir, 'PluginBenchmark')
    build_dir = os.path.join(project_dir, 'build')
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    buildConfig(script_dir, project_dir, cuda_path)
    print_green("Build Benchmark Plugin...")
    compileProjects(project_dir, "Debug")
    compileProjects(project_dir, "Release")


def build_unity(script_dir, unity_bin):
    unity_dir = os.path.join(script_dir, 'UnityBenchmark')
    print_green(
        "Build Unity project with OpenGL... It might take a few seconds...")
    unity_build_dir_gl = os.path.join(script_dir, 'build', 'UnityBenchmark-GL')
    os.makedirs(unity_build_dir_gl, exist_ok=True)
    subprocess.run([unity_bin, '-projectPath', unity_dir, '-buildTarget', 'Win64', '-executeMethod',
                    'BuildScript.BuildGL', '-logFile', '-', '-quit', '-batchmode'])
    source_gl_build_dir = os.path.join(unity_dir, 'build', 'GLBuild')
    if os.path.exists(source_gl_build_dir):
        for item in os.listdir(source_gl_build_dir):
            s = os.path.join(source_gl_build_dir, item)
            d = os.path.join(unity_build_dir_gl, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    print_green(
        "Build Unity project with DirectX11... It might take a few seconds...")
    unity_build_dir_dx11 = os.path.join(
        script_dir, 'build', 'UnityBenchmark-DX11')
    os.makedirs(unity_build_dir_dx11, exist_ok=True)
    subprocess.run([unity_bin, '-projectPath', unity_dir, '-buildTarget', 'Win64', '-executeMethod',
                    'BuildScript.BuildDX11', '-logFile', '-', '-quit', '-batchmode'])
    source_dx11_build_dir = os.path.join(unity_dir, 'build', 'DX11Build')
    if os.path.exists(source_dx11_build_dir):
        for item in os.listdir(source_dx11_build_dir):
            s = os.path.join(source_dx11_build_dir, item)
            d = os.path.join(unity_build_dir_dx11, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)


def benchmark(script_dir, num_tests, num_samples, batchmode):
    create_config_file(script_dir, num_samples)
    for i in range(num_tests):
        if batchmode:
            print_green(
                "Launch benchmark OpenGL in batchmode... It might takes several minutes")
        else:
            print_green("Launch benchmark OpenGL...")
        benchmark_gl_path = os.path.join(
            script_dir, 'build', 'UnityBenchmark-GL', 'UnityBenchmark.exe')
        if os.path.exists(benchmark_gl_path):
            benchmark_gl = subprocess.run(
                [benchmark_gl_path, '-batchmode' if batchmode else ''])
            if benchmark_gl.returncode == 0:
                results_dir_gl = os.path.join(
                    script_dir, 'Results', f'UnityBenchmark-GL-{i}')
                os.makedirs(results_dir_gl, exist_ok=True)
                for file in os.listdir(os.path.join(script_dir, 'build', 'UnityBenchmark-GL', 'UnityBenchmark_Data')):
                    if file.endswith('.csv'):
                        shutil.copy(os.path.join(script_dir, 'build', 'UnityBenchmark-GL', 'UnityBenchmark_Data', file),
                                    results_dir_gl)
        else:
            print_red(
                f"Error: {benchmark_gl_path} does not exist. Build the unity project before launching benchmark.")
            return

        if batchmode:
            print_green(
                "Launch benchmark DirectX11 in batchmode... It might takes several minutes")
        else:
            print_green("Launch benchmark DirectX11...")
        benchmark_dx11_path = os.path.join(
            script_dir, 'build', 'UnityBenchmark-DX11', 'UnityBenchmark.exe')
        if os.path.exists(benchmark_dx11_path):
            benchmark_dx11 = subprocess.run(
                [benchmark_dx11_path, '-batchmode' if batchmode else ''])
            if benchmark_dx11.returncode == 0:
                results_dir_dx11 = os.path.join(
                    script_dir, 'Results', f'UnityBenchmark-DX11-{i}')
                os.makedirs(results_dir_dx11, exist_ok=True)
                for file in os.listdir(os.path.join(script_dir, 'build', 'UnityBenchmark-DX11', 'UnityBenchmark_Data')):
                    if file.endswith('.csv'):
                        shutil.copy(os.path.join(script_dir, 'build', 'UnityBenchmark-DX11', 'UnityBenchmark_Data', file),
                                    results_dir_dx11)
        else:
            print_red(
                f"Error: {benchmark_dx11_path} does not exist. Build the unity project before launching benchmark.")
            return


def draw(script_dir):
    print_green("Create and save the graph of comparison")
    results_dir = os.path.join(script_dir, 'Results')
    if not os.path.exists(results_dir):
        print_red(
            "Error: Results directory does not exist. Run the benchmark first.")
        return

    run_dirs = [d for d in os.listdir(results_dir) if d.startswith(
        'UnityBenchmark-GL-') or d.startswith('UnityBenchmark-DX11-')]
    if not run_dirs:
        print_red("Error: No benchmark result directories found.")
        return

    for i, run_dir in enumerate(run_dirs, start=1):
        gl_result_dir = os.path.join(results_dir, f'UnityBenchmark-GL-{i-1}')
        dx11_result_dir = os.path.join(
            results_dir, f'UnityBenchmark-DX11-{i-1}')

        if os.path.exists(gl_result_dir) and os.path.exists(dx11_result_dir):
            gl_csv = os.path.join(gl_result_dir, 'ProfilingResults.csv')
            dx11_csv = os.path.join(dx11_result_dir, 'ProfilingResults.csv')
            # Use GL-0 or DX11-0 for CUDA
            cuda_csv = os.path.join(gl_result_dir, 'ProfilingResults.csv')

            title_arg = f"Performance Comparison - Run {i}"

            draw_args = ['python', os.path.join(
                script_dir, 'Results', 'main.py'), '-igl', gl_csv, '-idx', dx11_csv, '-icu', cuda_csv, '-t', title_arg, '-s', 'False', '-o', results_dir]
            subprocess.run(draw_args)
        else:
            print_red(
                f"Error: One or more result directories do not exist for run {i}.")


def main():
    parser = argparse.ArgumentParser(description='Build configuration script.')
    parser.add_argument("-c", "--cuda-path", default="C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v12.3",
                        help='Path to the CUDA toolkit (default: C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v12.3)')
    parser.add_argument("-u", "--unityBin", type=str, default="C://Program Files//Unity//Hub//Editor//2022.3.8f1//Editor//Unity.exe",
                        help='Path to the Unity executable.')
    parser.add_argument("-p", "--plugin", type=str2bool, default=True,
                        help='If true will configure and build the plugin benchmark, otherwise it won\'t')
    parser.add_argument("-pa", "--package-interop", type=str2bool, default=True,
                        help='If true will package InteropUnityCUDA, otherwise it won\'t')
    parser.add_argument("-bu", "--build-unity", type=str2bool, default=True,
                        help='If true will build unity benchmark project, otherwise it won\'t')
    parser.add_argument("-b", "--benchmark", type=str2bool, default=True,
                        help='If true will benchmark by launching unity project, otherwise it won\'t')
    parser.add_argument("-d", "--draw", type=str2bool, default=True,
                        help='If true will draw the graph of comparison, otherwise it won\'t')
    parser.add_argument("-ba", "--batchmode", type=str2bool, default=False,
                        help='If true will display the benchmark, otherwise it won\'t')
    parser.add_argument("-n", "--number-tests", type=int, default=1,
                        help='Number of times the benchmark should be launched')
    parser.add_argument("-ns", "--number-samples", type=int, default=1000,
                        help='Number of samples per size to update the Config.json')
    args = parser.parse_args()

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.package_interop:
        package_interop(script_dir)

    if args.plugin:
        generate_compile_plugin(script_dir, args.cuda_path)

    if args.build_unity:
        build_unity(script_dir, args.unityBin)

    if args.benchmark:
        benchmark(script_dir, args.number_tests,
                  args.number_samples, args.batchmode)

    if args.draw:
        draw(script_dir)


if __name__ == '__main__':
    main()

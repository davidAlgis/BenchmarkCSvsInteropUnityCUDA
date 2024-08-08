import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Define the new colors
GL_COLOR = (0.41, 0.71, 0.27)
DX11_COLOR = (0.85, 0.31, 0.29)
CUDA_COLOR = (0.12, 0.61, 0.73)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Set up argument parser
parser = argparse.ArgumentParser(description='Plot execution times from CSV.')
parser.add_argument('-igl', '--input-gl', type=str,
                    default='ProfilingResultsOpenGL.csv', help='Input OpenGL CSV file path')
parser.add_argument('-idx', '--input-dx11', type=str,
                    default='ProfilingResultsDX11.csv', help='Input DirectX 11 CSV file path')
parser.add_argument('-icu', '--input-cuda', type=str,
                    default='ProfilingResultsCUDA.csv', help='Input CUDA CSV file path')
parser.add_argument('-o', '--output', type=str, help='Output directory path')
parser.add_argument('-t', '--title', type=str,
                    default='Execution Sum Vector', help='Title of the graph')
parser.add_argument('-s', '--show', type=str2bool,
                    default='false', help='If true will show the graph, otherwise it won\'t')
parser.add_argument('-r', '--resolution', type=int, default=300,
                    help='Resolution of the output image in dpi (dots per inch)')
parser.add_argument('-x', '--xlim', type=str, default=None,
                    help='X-axis limits in the format "min,max"')
parser.add_argument('-y', '--ylim', type=str, default=None,
                    help='Y-axis limits in the format "min,max"')
parser.add_argument('-xt', '--xticks', type=str2bool,
                    default='false', help='If true will set x-ticks to correspond to the values in ArraySize')
args = parser.parse_args()

# Check if the input files exist
if not os.path.isfile(args.input_gl):
    raise FileNotFoundError(f"The file {args.input_gl} does not exist.")
if not os.path.isfile(args.input_dx11):
    raise FileNotFoundError(f"The file {args.input_dx11} does not exist.")
if not os.path.isfile(args.input_cuda):
    raise FileNotFoundError(f"The file {args.input_cuda} does not exist.")

# Create the output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

# Read the CSV files
gl_data = pd.read_csv(args.input_gl, delimiter=';')
dx11_data = pd.read_csv(args.input_dx11, delimiter=';')
cuda_data = pd.read_csv(args.input_cuda, delimiter=';')

# Convert the string values with commas to float values
gl_data['AverageExecutionTimeCS'] = gl_data['AverageExecutionTimeCS'].str.replace(
    ',', '.').astype(float)
gl_data['StandardDeviationCS'] = gl_data['StandardDeviationCS'].str.replace(
    ',', '.').astype(float)
dx11_data['AverageExecutionTimeCS'] = dx11_data['AverageExecutionTimeCS'].str.replace(
    ',', '.').astype(float)
dx11_data['StandardDeviationCS'] = dx11_data['StandardDeviationCS'].str.replace(
    ',', '.').astype(float)
cuda_data['AverageExecutionTimeCUDA'] = cuda_data['AverageExecutionTimeCUDA'].str.replace(
    ',', '.').astype(float)
cuda_data['StandardDeviationCUDA'] = cuda_data['StandardDeviationCUDA'].str.replace(
    ',', '.').astype(float)

# Plot the data with error bars
plt.figure(figsize=(10, 6), dpi=args.resolution)
plt.errorbar(gl_data['ArraySize'], gl_data['AverageExecutionTimeCS'], yerr=gl_data['StandardDeviationCS'],
             label='Average Execution Time OpenGL', marker='.', capsize=3, linestyle='-', color=GL_COLOR, markersize=3)
plt.errorbar(dx11_data['ArraySize'], dx11_data['AverageExecutionTimeCS'], yerr=dx11_data['StandardDeviationCS'],
             label='Average Execution Time DirectX 11', marker='.', capsize=3, linestyle='-', color=DX11_COLOR, markersize=3)
plt.errorbar(cuda_data['ArraySize'], cuda_data['AverageExecutionTimeCUDA'], yerr=cuda_data['StandardDeviationCUDA'],
             label='Average Execution Time CUDA', marker='.', capsize=3, linestyle='-', color=CUDA_COLOR, markersize=3)
plt.xlabel('Array Size')
plt.ylabel('Average Execution Time (ms)')
plt.title(args.title)
plt.legend()
plt.grid(True)
plt.xscale('log')

# Set the x-axis and y-axis limits if specified
if args.xlim:
    try:
        xmin, xmax = map(float, args.xlim.split(','))
        plt.xlim(xmin, xmax)
    except ValueError:
        print(f"Error: Invalid x-axis limits format '{args.xlim}'")
if args.ylim:
    try:
        ymin, ymax = map(float, args.ylim.split(','))
        plt.ylim(ymin, ymax)
    except ValueError:
        print(f"Error: Invalid y-axis limits format '{args.ylim}'")

# Set x-ticks to correspond to the values in 'ArraySize' if specified
if args.xticks:
    xticks = sorted(set(gl_data['ArraySize']) | set(
        dx11_data['ArraySize']) | set(cuda_data['ArraySize']))
    plt.xticks(xticks, xticks)

# Define the base output file name using the provided title
base_output_file_name = f'ProfilingResult-{args.title}'
# Ensure the output file name is unique
i = 1
output_file_name = f'{base_output_file_name}.png'
while os.path.isfile(output_file_name):
    output_file_name = f'{base_output_file_name}-{i}.png'
    i += 1
out = os.path.join(args.output, output_file_name)

# Save the plot with the unique file name
plt.savefig(out, dpi=args.resolution)
if args.show:
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Define the new colors
GL_COLOR = (0.41, 0.71, 0.27)
DX11_COLOR = (1, 0.43, 0.34)
CUDA_COLOR = (0.12, 0.61, 0.73)
CPU_COLOR = (1, 0.71, 0)  # Updated CPU color to orange
font_size_main_title = 22
font_size_title = 20
font_size_ticks = 18


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
parser.add_argument('-igl',
                    '--input-gl',
                    type=str,
                    default='ProfilingResultsOpenGL.csv',
                    help='Input OpenGL CSV file path')
parser.add_argument('-idx',
                    '--input-dx11',
                    type=str,
                    default='ProfilingResultsDX11.csv',
                    help='Input DirectX 11 CSV file path')
parser.add_argument('-icu',
                    '--input-cuda',
                    type=str,
                    default='ProfilingResultsCUDA.csv',
                    help='Input CUDA CSV file path')
parser.add_argument('-icp',
                    '--input-cpu',
                    type=str,
                    default='ProfilingResultsCPU.csv',
                    help='Input CPU CSV file path')  # New argument for CPU
parser.add_argument('-o',
                    '--output',
                    type=str,
                    default="Result",
                    help='Output directory path')
parser.add_argument('-t',
                    '--title',
                    type=str,
                    default='Execution Sum Vector',
                    help='Title of the graph')
parser.add_argument('-s',
                    '--show',
                    type=str2bool,
                    default='false',
                    help='If true will show the graph, otherwise it won\'t')
parser.add_argument(
    '-r',
    '--resolution',
    type=int,
    default=300,
    help='Resolution of the output image in dpi (dots per inch)')
parser.add_argument('-x',
                    '--xlim',
                    type=str,
                    default=None,
                    help='X-axis limits in the format "min,max"')
parser.add_argument('-y',
                    '--ylim',
                    type=str,
                    default=None,
                    help='Y-axis limits in the format "min,max"')
parser.add_argument(
    '-xt',
    '--xticks',
    type=str2bool,
    default='false',
    help='If true will set x-ticks to correspond to the values in ArraySize')
parser.add_argument('-l',
                    '--legend',
                    type=str2bool,
                    default='true',
                    help='If true will add the legend, otherwise it won\'t')
args = parser.parse_args()

# Check if the input files exist
missing_files = []
for file_arg in ['input_gl', 'input_dx11', 'input_cuda', 'input_cpu']:
    file_path = getattr(args, file_arg)
    if not os.path.isfile(file_path):
        missing_files.append(file_path)

if missing_files:
    missing = ', '.join(missing_files)
    raise FileNotFoundError(f"The following file(s) do not exist: {missing}")

# Create the output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

# Read the CSV files
gl_data = pd.read_csv(args.input_gl, delimiter=';')
dx11_data = pd.read_csv(args.input_dx11, delimiter=';')
cuda_data = pd.read_csv(args.input_cuda, delimiter=';')
cpu_data = pd.read_csv(args.input_cpu, delimiter=';')  # Read CPU data


# Function to replace commas and convert to float
def convert_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df


# Convert relevant columns in each DataFrame
gl_data = convert_columns(gl_data,
                          ['AverageExecutionTimeCS', 'StandardDeviationCS'])
dx11_data = convert_columns(dx11_data,
                            ['AverageExecutionTimeCS', 'StandardDeviationCS'])
cuda_data = convert_columns(
    cuda_data, ['AverageExecutionTimeCUDA', 'StandardDeviationCUDA'])
cpu_data = convert_columns(
    cpu_data,
    ['AverageExecutionTimeCPU', 'StandardDeviationCPU'])  # Convert CPU columns

# Determine which metrics to plot based on average execution times
plot_gl = not gl_data['AverageExecutionTimeCS'].eq(0).all()
plot_dx11 = not dx11_data['AverageExecutionTimeCS'].eq(0).all()
plot_cuda = not cuda_data['AverageExecutionTimeCUDA'].eq(0).all()
plot_cpu = not cpu_data['AverageExecutionTimeCPU'].eq(0).all()

# Collect lines and labels for legend
lines = []
labels = []

# Plot the data with error bars
plt.figure(figsize=(12, 8),
           dpi=args.resolution)  # Increased figure size for better visibility

# Plot OpenGL Compute Shader metrics if applicable
if plot_gl:
    line_gl = plt.errorbar(gl_data['ArraySize'],
                           gl_data['AverageExecutionTimeCS'],
                           yerr=gl_data['StandardDeviationCS'],
                           label='Average Execution Time OpenGL CS',
                           marker='o',
                           capsize=3,
                           linestyle='dashed',
                           color=GL_COLOR,
                           markersize=5)
    lines.append(line_gl)
    labels.append('Average Execution Time OpenGL CS')
else:
    print("Skipping OpenGL CS plot: All average execution times are zero.")

# Plot DirectX 11 Compute Shader metrics if applicable
if plot_dx11:
    line_dx11 = plt.errorbar(dx11_data['ArraySize'],
                             dx11_data['AverageExecutionTimeCS'],
                             yerr=dx11_data['StandardDeviationCS'],
                             label='Average Execution Time DirectX 11 CS',
                             marker='s',
                             capsize=3,
                             linestyle='dashdot',
                             color=DX11_COLOR,
                             markersize=5)
    lines.append(line_dx11)
    labels.append('Average Execution Time DirectX 11 CS')
else:
    print("Skipping DirectX 11 CS plot: All average execution times are zero.")

# Plot CUDA metrics if applicable
if plot_cuda:
    line_cuda = plt.errorbar(cuda_data['ArraySize'],
                             cuda_data['AverageExecutionTimeCUDA'],
                             yerr=cuda_data['StandardDeviationCUDA'],
                             label='Average Execution Time CUDA',
                             marker='^',
                             capsize=3,
                             linestyle='-',
                             color=CUDA_COLOR,
                             markersize=5)
    lines.append(line_cuda)
    labels.append('Average Execution Time CUDA')
else:
    print("Skipping CUDA plot: All average execution times are zero.")

# Plot CPU metrics if applicable
if plot_cpu:
    line_cpu = plt.errorbar(cpu_data['ArraySize'],
                            cpu_data['AverageExecutionTimeCPU'],
                            yerr=cpu_data['StandardDeviationCPU'],
                            label='Average Execution Time CPU',
                            marker='D',
                            capsize=3,
                            linestyle=':',
                            color=CPU_COLOR,
                            markersize=5)
    lines.append(line_cpu)
    labels.append('Average Execution Time CPU')
else:
    print("Skipping CPU plot: All average execution times are zero.")

# Check if at least one metric is plotted
if not lines:
    print("Error: No metrics to plot. All average execution times are zero.")
    exit(1)

# Set labels and title
plt.xlabel('Array Size', fontsize=font_size_title, labelpad=10)
plt.ylabel('Average Execution Time (ms)',
           fontsize=font_size_title,
           labelpad=10)
plt.title(args.title, fontsize=font_size_main_title, pad=20)
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Set log scale for x-axis
plt.xscale('log')

# Optional: Set log scale for y-axis if needed
# plt.yscale('log')

# Handle legend
if args.legend:
    plt.legend(fontsize=font_size_ticks)
else:
    if (len(lines) == 4):
        # Create the legend separately
        fig_legend = plt.figure(figsize=(5, 2))
        fig_legend.legend(handles=lines,
                          labels=labels,
                          loc='center',
                          fontsize=font_size_ticks)
        legend_out = os.path.join(args.output, 'Legend.png')
        fig_legend.savefig(legend_out,
                           dpi=args.resolution,
                           bbox_inches='tight')
        plt.close(fig_legend)

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
else:
    # If ylim is not specified, set it based on the max value of all plotted metrics
    max_values = []
    if plot_gl:
        max_values.append(gl_data['AverageExecutionTimeCS'].max())
    if plot_dx11:
        max_values.append(dx11_data['AverageExecutionTimeCS'].max())
    if plot_cuda:
        max_values.append(cuda_data['AverageExecutionTimeCUDA'].max())

    if max_values:
        overall_max = max(max_values)
        padding = overall_max * 0.05  # 5% padding
        plt.ylim(0, overall_max + padding)
    else:
        # Fallback in case no max_values are found (shouldn't happen as we checked above)
        plt.ylim(0, 1)
        print("Y-axis limits set to (0, 1) as a fallback.")

# Set x-ticks to correspond to the values in 'ArraySize' if specified
if args.xticks:
    xticks = sorted(
        set(gl_data['ArraySize']) | set(dx11_data['ArraySize'])
        | set(cuda_data['ArraySize']) | set(cpu_data['ArraySize']))
    plt.xticks(xticks, xticks)

# Set y-ticks font size
plt.yticks(fontsize=font_size_ticks)
plt.xticks(fontsize=font_size_ticks)

# Adjust the layout to increase space between figure and titles
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.95)

# Define the base output file name using the provided title
name = args.title.replace(' ', '')
base_output_file_name = f'ProfilingResult-{name}'

# Ensure the output file name is unique
i = 1
output_file_name = f'{base_output_file_name}.png'
while os.path.isfile(os.path.join(args.output, output_file_name)):
    output_file_name = f'{base_output_file_name}-{i}.png'
    i += 1
out = os.path.join(args.output, output_file_name)

# Save the plot with the unique file name
plt.savefig(out, dpi=args.resolution, bbox_inches='tight')
if args.show:
    plt.show()

print(f"Plot saved to {out}")

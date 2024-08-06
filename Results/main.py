import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


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
parser.add_argument('-i', '--input', type=str,
                    default='ProfilingResults.csv', help='Input CSV file path')
parser.add_argument('-o', '--output', type=str, help='Output directory path')
parser.add_argument('-t', '--title', type=str,
                    default='Execution Sum Vector', help='Title of the graph')
parser.add_argument('-s', '--show', type=str2bool,
                    default='false', help='If true will show the graph, otherwise it won\'t')
args = parser.parse_args()

# Check if the input file exists
if not os.path.isfile(args.input):
    raise FileNotFoundError(f"The file {args.input} does not exist.")

# Read the CSV file
file_path = args.input
data = pd.read_csv(file_path, delimiter=';')

# Convert the string values with commas to float values
data['AverageExecutionTimeCS'] = data['AverageExecutionTimeCS'].str.replace(
    ',', '.').astype(float)
data['StandardDeviationCS'] = data['StandardDeviationCS'].str.replace(
    ',', '.').astype(float)
data['AverageExecutionTimeCUDA'] = data['AverageExecutionTimeCUDA'].str.replace(
    ',', '.').astype(float)
data['StandardDeviationCUDA'] = data['StandardDeviationCUDA'].str.replace(
    ',', '.').astype(float)

# Plot the data with error bars
plt.figure(figsize=(10, 6))

plt.errorbar(data['ArraySize'], data['AverageExecutionTimeCS'], yerr=data['StandardDeviationCS'],
             label='Average Execution Time Compute Shader', marker='o', capsize=5, linestyle='-')
plt.errorbar(data['ArraySize'], data['AverageExecutionTimeCUDA'], yerr=data['StandardDeviationCUDA'],
             label='Average Execution Time CUDA', marker='o', capsize=5, linestyle='-')

plt.xlabel('Array Size')
plt.ylabel('Average Execution Time (ms)')
plt.title(args.title)
plt.legend()
plt.grid(True)
plt.xscale('log')

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
plt.savefig(out)
if (args.show):
    plt.show()

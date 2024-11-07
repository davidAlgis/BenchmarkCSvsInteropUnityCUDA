import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
import math

# Define the colors
GL_COLOR = (0.41, 0.71, 0.27)
DX11_COLOR = (1, 0.43, 0.34)
CUDA_COLOR = (0.12, 0.61, 0.73)
CPU_COLOR = (1, 0.71, 0)
font_size_main_title = 22
font_size_title = 20
font_size_ticks = 18


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot execution times and speedup from CSV.')
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
                        help='Input CPU CSV file path')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default="Result",
                        help='Output directory path')
    parser.add_argument('-t',
                        '--title',
                        type=str,
                        default='Execution and Speedup Metrics',
                        help='Base title for the graphs')
    parser.add_argument('-s',
                        '--show',
                        type=str2bool,
                        default='false',
                        help='If true, will show the graphs')
    parser.add_argument('-r',
                        '--resolution',
                        type=int,
                        default=300,
                        help='Resolution of the output images in dpi')
    parser.add_argument(
        '-xa',
        '--xlimAverage',
        type=str,
        default=None,
        help='X-axis limits for average plot in the format "min,max"')
    parser.add_argument(
        '-ya',
        '--ylimAverage',
        type=str,
        default=None,
        help='Y-axis limits for average plot in the format "min,max"')
    parser.add_argument(
        '-xs',
        '--xlimSpeedup',
        type=str,
        default=None,
        help='X-axis limits for speedup plot in the format "min,max"')
    parser.add_argument(
        '-ys',
        '--ylimSpeedup',
        type=str,
        default=None,
        help='Y-axis limits for speedup plot in the format "min,max"')
    parser.add_argument(
        '-xt',
        '--xticks',
        type=str2bool,
        default='false',
        help='If true, set x-ticks to correspond to ArraySize values')
    parser.add_argument('-l',
                        '--legend',
                        type=str2bool,
                        default='true',
                        help='If true, add the legend')
    return parser.parse_args()


def check_input_files(args):
    """Check if input files exist."""
    missing_files = []
    for file_arg in ['input_gl', 'input_dx11', 'input_cuda', 'input_cpu']:
        file_path = getattr(args, file_arg)
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
    if missing_files:
        missing = ', '.join(missing_files)
        raise FileNotFoundError(
            f"The following file(s) do not exist: {missing}")


def load_and_preprocess_data(file_path, avg_col, std_col):
    """Load and preprocess CSV data."""
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        sys.exit(1)
    df = df.copy()  # Avoid SettingWithCopyWarning
    df = convert_columns(df, [avg_col, std_col])
    return df


def convert_columns(df, columns):
    """Convert specified columns to float after replacing commas."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                print(f"Error: Column '{col}' contains non-numeric values.")
                df[col] = 0.0
    return df


def calculate_speedup(cuda_data, metric_data, avg_col_cuda, std_col_cuda,
                      avg_col_metric, std_col_metric):
    """Calculate speedup and errors."""
    speedup = []
    speedup_err = []
    array_sizes = []
    for idx in range(len(metric_data)):
        cuda_avg = cuda_data[avg_col_cuda].iloc[idx]
        cuda_std = cuda_data[std_col_cuda].iloc[idx]
        metric_avg = metric_data[avg_col_metric].iloc[idx]
        metric_std = metric_data[std_col_metric].iloc[idx]
        s, err = _calculate_speedup_values(cuda_avg, cuda_std, metric_avg,
                                           metric_std)
        if s is not None:
            speedup.append(s)
            speedup_err.append(err if err is not None else 0)
            array_sizes.append(metric_data['ArraySize'].iloc[idx])
    return array_sizes, speedup, speedup_err


def _calculate_speedup_values(cuda_avg, cuda_std, metric_avg, metric_std):
    """Helper function to calculate speedup and its error."""
    if metric_avg == 0 or cuda_avg == 0:
        return None, None
    speedup = cuda_avg / metric_avg
    try:
        dz = speedup * math.sqrt((cuda_std / cuda_avg)**2 +
                                 (metric_std / metric_avg)**2)
    except ZeroDivisionError:
        dz = None
    return speedup, dz


def plot_metrics(args,
                 data_dict,
                 title_suffix,
                 ylabel,
                 output_filename,
                 is_speedup=False,
                 xlim=None,
                 ylim=None):
    """Plot metrics (average times or speedups)."""
    plt.figure(figsize=(12, 8), dpi=args.resolution)
    lines = []
    labels = []
    max_values = []

    for key, value in data_dict.items():
        if not value['plot']:
            continue  # Skip if the metric should not be plotted

        # Ensure that 'y' is not empty before appending the maximum value
        if isinstance(value['y'], pd.Series):
            if not value['y'].empty:
                if (key != 'cpu'):
                    max_values.append(value['y'].max())
        elif isinstance(value['y'], list):
            if value['y']:  # Check if the list is non-empty
                if (key != 'cpu'):
                    max_values.append(max(value['y']))

        # Plot the metric
        line = plt.errorbar(value['x'],
                            value['y'],
                            yerr=value['yerr'],
                            label=value['label'],
                            marker=value['marker'],
                            capsize=3,
                            linestyle=value['linestyle'],
                            color=value['color'],
                            markersize=5)
        lines.append(line)
        labels.append(value['label'])

    if not lines:
        print(f"Error: No valid data to plot for {title_suffix}.")
        sys.exit(1)

    # Set labels and title
    plt.xlabel('Array Size', fontsize=font_size_title, labelpad=10)
    plt.ylabel(ylabel, fontsize=font_size_title, labelpad=10)
    plt.title(f"{args.title} - {title_suffix}",
              fontsize=font_size_main_title,
              pad=20)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xscale('log')

    # Handle legend
    if args.legend:
        plt.legend(fontsize=font_size_ticks)

    # Set axis limits
    if xlim:
        try:
            xmin, xmax = map(float, xlim.split(','))
            plt.xlim(xmin, xmax)
        except ValueError:
            print(f"Error: Invalid x-axis limits format '{xlim}'")
    if ylim:
        try:
            ymin, ymax = map(float, ylim.split(','))
            plt.ylim(ymin, ymax)
        except ValueError:
            print(f"Error: Invalid y-axis limits format '{ylim}'")
    else:
        if max_values:
            overall_max = max(max_values)
            padding = overall_max * 0.05  # 5% padding
            plt.ylim(0, overall_max + padding)
        else:
            plt.ylim(0, 1)
            print(
                f"Y-axis limits set to (0, 1) as a fallback for '{title_suffix}'."
            )

    # Set x-ticks if specified
    if args.xticks:
        xticks = sorted(set().union(*(value['x']
                                      for value in data_dict.values()
                                      if value['plot'])))
        plt.xticks(xticks, xticks)

    # Set font sizes
    plt.yticks(fontsize=font_size_ticks)
    plt.xticks(fontsize=font_size_ticks)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.95)

    # Save plot
    save_plot(args, output_filename)
    if args.show:
        plt.show()
    plt.close()


def save_plot(args, base_filename):
    """Save the plot to a file."""
    base_output_file_name = base_filename
    i = 1
    output_file_name = f'{base_output_file_name}.png'
    while os.path.isfile(os.path.join(args.output, output_file_name)):
        output_file_name = f'{base_output_file_name}-{i}.png'
        i += 1
    out = os.path.join(args.output, output_file_name)
    plt.savefig(out, dpi=args.resolution, bbox_inches='tight')
    print(f"Plot saved to {out}")


def main():
    args = parse_arguments()
    check_input_files(args)
    os.makedirs(args.output, exist_ok=True)
    # Load and preprocess data
    gl_data = load_and_preprocess_data(args.input_gl, 'AverageExecutionTimeCS',
                                       'StandardDeviationCS')
    dx11_data = load_and_preprocess_data(args.input_dx11,
                                         'AverageExecutionTimeCS',
                                         'StandardDeviationCS')
    cuda_data = load_and_preprocess_data(args.input_cuda,
                                         'AverageExecutionTimeCUDA',
                                         'StandardDeviationCUDA')
    cpu_data = load_and_preprocess_data(args.input_cpu,
                                        'AverageExecutionTimeCPU',
                                        'StandardDeviationCPU')
    # Determine which metrics to plot
    plot_gl = not gl_data['AverageExecutionTimeCS'].eq(0).all()
    plot_dx11 = not dx11_data['AverageExecutionTimeCS'].eq(0).all()
    plot_cuda = not cuda_data['AverageExecutionTimeCUDA'].eq(0).all()
    plot_cpu = not cpu_data['AverageExecutionTimeCPU'].eq(0).all()
    if not plot_cuda:
        print("Error: CUDA metrics are not plotted. Cannot compute speedup.")
        sys.exit(1)
    # Prepare data for average execution times
    avg_data_dict = {}
    if plot_gl:
        avg_data_dict['gl'] = {
            'plot': plot_gl,
            'x': gl_data['ArraySize'],
            'y': gl_data['AverageExecutionTimeCS'],
            'yerr': gl_data['StandardDeviationCS'],
            'label': 'Average Execution Time OpenGL CS',
            'marker': 'o',
            'linestyle': 'dashed',
            'color': GL_COLOR
        }
    if plot_dx11:
        avg_data_dict['dx11'] = {
            'plot': plot_dx11,
            'x': dx11_data['ArraySize'],
            'y': dx11_data['AverageExecutionTimeCS'],
            'yerr': dx11_data['StandardDeviationCS'],
            'label': 'Average Execution Time DirectX 11 CS',
            'marker': 's',
            'linestyle': 'dashdot',
            'color': DX11_COLOR
        }
    if plot_cuda:
        avg_data_dict['cuda'] = {
            'plot': plot_cuda,
            'x': cuda_data['ArraySize'],
            'y': cuda_data['AverageExecutionTimeCUDA'],
            'yerr': cuda_data['StandardDeviationCUDA'],
            'label': 'Average Execution Time CUDA',
            'marker': '^',
            'linestyle': '-',
            'color': CUDA_COLOR
        }
    if plot_cpu:
        avg_data_dict['cpu'] = {
            'plot': plot_cpu,
            'x': cpu_data['ArraySize'],
            'y': cpu_data['AverageExecutionTimeCPU'],
            'yerr': cpu_data['StandardDeviationCPU'],
            'label': 'Average Execution Time CPU',
            'marker': 'D',
            'linestyle': ':',
            'color': CPU_COLOR
        }
    # Plot average execution times
    name = args.title.replace(' ', '')
    execution_plot_filename = f'ProfilingResult-{name}'
    plot_metrics(args,
                 avg_data_dict,
                 'Average Execution Time',
                 'Average Execution Time (ms)',
                 execution_plot_filename,
                 is_speedup=False,
                 xlim=args.xlimAverage,
                 ylim=args.ylimAverage)
    # Prepare data for speedup
    speedup_data_dict = {}
    if plot_gl:
        array_sizes_gl, speedup_gl, speedup_gl_err = calculate_speedup(
            cuda_data, gl_data, 'AverageExecutionTimeCUDA',
            'StandardDeviationCUDA', 'AverageExecutionTimeCS',
            'StandardDeviationCS')
        if speedup_gl:
            speedup_data_dict['gl'] = {
                'plot': True,
                'x': array_sizes_gl,
                'y': speedup_gl,
                'yerr': speedup_gl_err,
                'label': 'Speedup OpenGL CS',
                'marker': 'o',
                'linestyle': 'dashed',
                'color': GL_COLOR
            }
    if plot_dx11:
        array_sizes_dx11, speedup_dx11, speedup_dx11_err = calculate_speedup(
            cuda_data, dx11_data, 'AverageExecutionTimeCUDA',
            'StandardDeviationCUDA', 'AverageExecutionTimeCS',
            'StandardDeviationCS')
        if speedup_dx11:
            speedup_data_dict['dx11'] = {
                'plot': True,
                'x': array_sizes_dx11,
                'y': speedup_dx11,
                'yerr': speedup_dx11_err,
                'label': 'Speedup DirectX 11 CS',
                'marker': 's',
                'linestyle': 'dashdot',
                'color': DX11_COLOR
            }
    if plot_cpu:
        array_sizes_cpu, speedup_cpu, speedup_cpu_err = calculate_speedup(
            cuda_data, cpu_data, 'AverageExecutionTimeCUDA',
            'StandardDeviationCUDA', 'AverageExecutionTimeCPU',
            'StandardDeviationCPU')
        if speedup_cpu:
            speedup_data_dict['cpu'] = {
                'plot': True,
                'x': array_sizes_cpu,
                'y': speedup_cpu,
                'yerr': speedup_cpu_err,
                'label': 'Speedup CPU',
                'marker': 'D',
                'linestyle': ':',
                'color': CPU_COLOR
            }
    if plot_cuda:
        array_sizes_cuda, speedup_cuda, speedup_cuda_err = calculate_speedup(
            cuda_data, cuda_data, 'AverageExecutionTimeCUDA',
            'StandardDeviationCUDA', 'AverageExecutionTimeCUDA',
            'StandardDeviationCUDA')
        if speedup_cpu:
            speedup_data_dict['cuda'] = {
                'plot': True,
                'x': array_sizes_cuda,
                'y': speedup_cuda,
                'yerr': speedup_cuda_err,
                'label': 'Speedup CUDA',
                'marker': 'D',
                'linestyle': ':',
                'color': CUDA_COLOR
            }
    # Plot speedup
    speedup_plot_filename = f'ProfilingSpeedup-{name}'
    plot_metrics(args,
                 speedup_data_dict,
                 'Speedup Compared to CUDA',
                 'Speedup Compared to CUDA',
                 speedup_plot_filename,
                 is_speedup=True,
                 xlim=args.xlimSpeedup,
                 ylim=args.ylimSpeedup)


if __name__ == '__main__':
    main()

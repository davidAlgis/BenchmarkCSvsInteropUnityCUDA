# Execution Time Plotter

This Python script reads the data from the CSV file generate by `UnityBenchmark` build and generates a plot comparing the average execution times for Compute Shader (CS) and CUDA. 

## Installation

Before running the script, you need to install the required Python libraries. You can install them using the provided `requirements.txt` file.

```
pip install -r requirements.txt
```

## Usage

Ensure you have Python 3.6 or newer installed on your system. Clone this repository or download the script and `requirements.txt` file. Install the required libraries as mentioned above. To use the script, run it from the command line with the desired options:

```
python main.py [options]
```

## Options

- `-i`, `--input <file>`: Specify the input CSV file path. If not specified, the script uses `ProfilingResults.csv` by default.

- `-t`, `--title <title>`: Specify the title of the graph. This title will also be used to help define the name of the output file. The default title is "Execution Sum Vector".

- `-h`, `--help`: Display help information showing all command-line options.

## Example

To generate a plot from a CSV file named `MyProfilingResults1-release.csv` with a custom title "My Custom Title", you would use:

```
python main.py -i MyProfilingResults1-release.csv -t "My Custom Title"
```

This will save the output file as `ProfilingResult-My Custom Title.png`. If a file with that name already exists, it will append `-1`, `-2`, etc., to the file name to ensure uniqueness.

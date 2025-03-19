# IT_PI: Information-theoretic Buckingham-Π theorem

This repository contains a command-line implementation of the Dimensional Analysis Tool, converted from the original Jupyter Notebook version. The tool uses mutual information to discover optimal dimensionless groups in physical datasets.

## Key Features

### 1. New Input/Output System

The implementation introduces a robust configuration-based input/output system:

- **Configuration-Based:** Uses YAML/JSON configuration files to control all aspects of the analysis
- **Multiple File Support:** Can analyze data from multiple sources simultaneously
- **Format Flexibility:** Supports both pickle (.pkl) and CSV file formats
- **Customizable Analysis:** Adjust calculation formulas, input columns, and dimensionless groups through config

### 2. Multiprocessing Support

Significant performance improvements through parallel processing:

- **Parallel Evaluation:** Uses CMA-ES optimization with parallel fitness evaluation
- **Configurable Processors:** Control the number of processors through the `n_procs` parameter

### 3. Enhanced Data Handling

- **Sample Size Control:** Limit the number of samples for faster analysis
- **Random Sampling:** Option to use random sampling or sequential data selection
- **Flexible Variable Selection:** Choose input variables and custom formulas

## Usage

### Basic Command

```bash
python run_analysis.py config_file.yaml
```

### Generate a Default Configuration

```bash
python input_system.py --generate-config my_config.yaml
```

## Configuration File

The YAML configuration file controls all aspects of the analysis:

```yaml
data:
  file_paths:
    - "./data/file1.csv"
    - "./data/file2.pkl"
  data_key: "unnormalized_inputs"  # Key for pickle files
  column_indices:
    y: 0
    U_1: 1
    # more mappings...
  sample_size: 1000
  random_sampling: true

analysis:
  output_calculation: "utau * y / nu"
  input_columns: [0, 1, 2, 4, 5, 6]
  variable_names: ["y", "U_1", "\\nu", "u_p", "U_2", "U_3"]
  dimension_matrix: 
    - [1, 1, 2, 1, 1, 1]
    - [0, -1, -1, -1, -1, -1]
  num_dimensionless_groups: 3

optimization:
  binning_bins: 30
  n_procs: 20  # Number of processors for parallel execution
  # additional optimization settings...

output:
  save_results: true
  results_dir: "./results"
  plot_results: true

```
See examples for annotated configuration files.

## Examples
Current examples (check ./examples/) include:
1. Turbulent boundary layer data with different adverse pressure gradients (Note that this example is just to help you understand the tool and the method, and the results are not accurate as data are severely downsampled)

## Converting Between File Formats

The tool in utils/ provides utility functions to convert between file formats:

## Results

The tool generates:

1. Pickle files with complete results
2. Human-readable text summaries
3. Visualizations of dimensionless groups
4. Combined plots showing relationships between groups

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- PyYAML
- pandas
- CMA-ES

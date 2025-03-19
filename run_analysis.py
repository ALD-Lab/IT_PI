"""
Script to run dimensional analysis using the input system.
Support for multiple data files.

Example:
python run_analysis.py ./examples/TBL/config_test_pkl.yaml

The results will be saved in the same directory as the configuration file.

"""

import sys
import os
import argparse
from input_system import DimensionalAnalysisInput
import numpy as np

# Import dimensional_analysis from the updated code
from Nondimlearning import dimensional_analysis

def run_analysis(config_file, data_files=None):
    """
    Run dimensional analysis using the specified configuration file.
    
    Parameters:
    -----------
    config_file : str
        Path to the configuration file
    data_files : list, optional
        List of paths to data files, overriding the config
    
    Returns:
    --------
    dict
        Results from the dimensional analysis
    """
    # Initialize the input system
    input_system = DimensionalAnalysisInput(config_file)
    
    # Override file paths if provided
    if data_files:
        input_system.config['data']['file_paths'] = data_files
    
    # Load data
    print("Loading data...")
    print(f"Using the following data files: {input_system.config['data']['file_paths']}")
    
    X, Y, variable_names, dimension_matrix, basis_matrices = input_system.load_data()
    print(f"Loaded data: X shape = {X.shape}, Y shape = {Y.shape}")
    print(f"Variable names: {variable_names}")
    print(f"Dimension matrix:\n{dimension_matrix}")
    
    if basis_matrices is not None:
        print(f"Using provided basis matrices:\n{basis_matrices}")
    
    # Get optimization parameters
    opt_params = input_system.get_optimization_params()
    
    # Run dimensional analysis
    print("\nRunning dimensional analysis...")
    results = dimensional_analysis(
        X, 
        Y, 
        dimension_matrix, 
        variable_names, 
        opt_params['num_dimensionless_groups'],
        basis_matrices=basis_matrices,
        n_procs=opt_params['n_procs'],
        cma_options=opt_params['cma_options'],
        binning_bins=opt_params['binning_bins'],
        k_nn=opt_params['k_nn'],
        initial_conditions=opt_params['initial_conditions']
    )
    
    # Save results
    print("\nSaving results...")
    input_system.save_results(results)
    
    # Print summary
    print("\nAnalysis Results:")
    print("----------------")
    print(f"Optimized MI: {results['optimized_MI']}")
    print("\nDimensionless Groups:")
    for i, label in enumerate(results["dimensionless_labels"]):
        print(f"Group {i+1}: {label}")

    # Plot results
    print("\nPlotting results...")
    input_system.plot_results(results, X, Y)
    
    return results

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Run Dimensional Analysis')
    parser.add_argument('config_file', type=str, help='Path to configuration file')
    parser.add_argument('--data', '-d', type=str, nargs='+', help='Path(s) to data file(s)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file '{args.config_file}' not found.")
        sys.exit(1)
    
    # Check if provided data files exist
    if args.data:
        for file_path in args.data:
            if not os.path.exists(file_path):
                print(f"Error: Data file '{file_path}' not found.")
                sys.exit(1)
    
    run_analysis(args.config_file, args.data)

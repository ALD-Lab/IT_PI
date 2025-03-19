import numpy as np
import yaml
import argparse
import os
import pickle as pkl
from pathlib import Path
import json
import sys
import numpy as np
from Nondimlearning import dimensional_analysis
import pandas as pd
import os

class DimensionalAnalysisInput:
    """
    A class to handle input configuration for dimensional analysis.
    """
    def __init__(self, config_file=None):
        """
        Initialize the input system with optional config file.
        
        Parameters:
        -----------
        config_file : str, optional
            Path to the configuration file (YAML or JSON)
        """
        self.config = {
            # Default configuration
            'data': {
                'file_paths': [],  # Now a list to hold multiple file paths
                'data_key': 'unnormalized_inputs',
                'column_indices': {
                    'y': 0,
                    'U_1': 1,
                    'nu': 2,
                    'utau': 3,
                    'up': 4,
                    'U_2': 5,
                    'U_3': 6,
                    'U_4': 7
                },
                'sample_size': None,  # Use all data by default
            },
            'analysis': {
                'output_calculation': 'utau * y / nu',  # Formula to calculate Y
                'input_columns': [0, 1, 3, 4],  # Columns to use for X
                'variable_names': ['y', 'u_1', '\\nu', 'u_p'],
                'dimension_matrix': [[1, 1, 2, -1], [0, -1, -1, -1]],
                'basis_matrices': None,  # Will be calculated if not provided
                'num_dimensionless_groups': 2
            },
            'optimization': {
                'binning_bins': 30,
                'k_nn': 5,
                'n_procs': 20,
                'cma_options': {
                    'bounds': [[-2, -2, -2, -2], [2, 2, 2, 2]],
                    'maxiter': 50000,
                    'tolx': 1e-4,
                    'tolfun': 1e-4,
                    'popsize': 300
                }
            },
            'output': {
                'save_results': True,
                'results_dir': './results',
                'plot_results': True
            }
        }
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """
        Load configuration from a file.
        
        Parameters:
        -----------
        config_file : str
            Path to the configuration file
        """
        file_ext = os.path.splitext(config_file)[1].lower()
        
        try:
            with open(config_file, 'r') as f:
                if file_ext == '.yaml' or file_ext == '.yml':
                    user_config = yaml.safe_load(f)
                elif file_ext == '.json':
                    user_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_ext}")
                
                # Update the configuration with user values
                self._update_dict(self.config, user_config)
                
                # Backward compatibility: If file_path exists, add it to file_paths
                if 'file_path' in self.config['data'] and self.config['data']['file_path']:
                    if not self.config['data']['file_paths']:
                        self.config['data']['file_paths'] = [self.config['data']['file_path']]
                    
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"Configuration file not found: {config_file}")
            sys.exit(1)
    
    def _update_dict(self, d, u):
        """
        Recursively update a dictionary with another dictionary.
        
        Parameters:
        -----------
        d : dict
            Dictionary to update
        u : dict
            Dictionary with update values
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v
    
    def save_config(self, output_file):
        """
        Save the current configuration to a file.
        
        Parameters:
        -----------
        output_file : str
            Path to save the configuration file
        """
        file_ext = os.path.splitext(output_file)[1].lower()
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            if file_ext == '.yaml' or file_ext == '.yml':
                yaml.dump(self.config, f, default_flow_style=False)
            elif file_ext == '.json':
                json.dump(self.config, f, indent=4)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
    
    def _load_pickle_file(self, file_path):
        """
        Load data from a pickle file.
        
        Parameters:
        -----------
        file_path : str
            Path to the pickle file.
            
        Returns:
        --------
        tuple
            Data array
        """
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
        
        # Extract the data using the specified key
        data_key = self.config['data']['data_key']
        if data_key not in data:
            raise KeyError(f"Data key '{data_key}' not found in the data file: {file_path}")
        
        data_unnormalized = data[data_key]
            
        return data_unnormalized

    def _load_csv_file(self, file_path):
        """
        Load data from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file.
            
        Returns:
        --------
        tuple
            Data array and weight (default 1.0 for CSV files)
        """
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Check if the CSV has a header row based on config
        has_header = self.config['data'].get('csv_has_header', True)
        if not has_header:
            # If no header, rename columns to numerical indices
            df.columns = list(range(len(df.columns)))
        
        # Convert DataFrame to numpy array
        data_unnormalized = df.values
        
        return data_unnormalized

    def load_data(self):
        """
        Load data from the specified files (pickle or CSV).
        
        Returns:
        --------
        tuple
            X input data, Y output data, variable names, dimension matrix, and basis matrices
        """
        file_paths = self.config['data']['file_paths']
        if not file_paths:
            raise ValueError("No data file paths specified")
        
        # Lists to store data from each file
        all_data_unnormalized = []
        
        try:
            # Load data from each file
            for file_path in file_paths:
                # Determine file type based on extension
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pkl':
                    data_unnormalized = self._load_pickle_file(file_path)
                elif file_extension == '.csv':
                    data_unnormalized = self._load_csv_file(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .pkl, .csv")
                
                all_data_unnormalized.append(data_unnormalized)
            
            # Simply append all data
            data_unnormalized = np.vstack(all_data_unnormalized)
            
            # Extract variables based on column indices
            col_indices = self.config['data']['column_indices']
            variables = {}
            for var_name, col_idx in col_indices.items():
                variables[var_name] = data_unnormalized[:, col_idx]
            
            # Calculate Y using the specified formula
            formula = self.config['analysis']['output_calculation']
            Y = eval(formula, {"__builtins__": {}}, variables)
            # Extract X based on input_columns
            input_cols = self.config['analysis']['input_columns']
            X = data_unnormalized[:, input_cols]
            
            # Apply sample size limit if specified
            sample_size = self.config['data']['sample_size']
            if sample_size and sample_size < len(X):
                # Random sampling option
                if self.config['data'].get('random_sampling', False):
                    indices = np.random.choice(len(X), sample_size, replace=False)
                    X = X[indices, :]
                    Y = Y[indices]
                else:
                    # Take first n samples
                    X = X[:sample_size, :]
                    Y = Y[:sample_size]
            
            # Ensure Y is properly shaped
            Y = np.array(Y).reshape(-1, 1)
            
            # Get other required parameters
            variable_names = self.config['analysis']['variable_names']
            dimension_matrix = np.array(self.config['analysis']['dimension_matrix'])
            
            # Get or calculate basis matrices
            basis_matrices = self.config['analysis']['basis_matrices']
            if basis_matrices:
                basis_matrices = np.array(basis_matrices)
            
            return X, Y, variable_names, dimension_matrix, basis_matrices
            
        except (FileNotFoundError, IOError, pd.errors.EmptyDataError) as e:
            print(f"Error loading data file: {e}")
            sys.exit(1) 

    def add_data_file(self, file_path):
        """
        Add a data file to the list of file paths.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        """
        if 'file_paths' not in self.config['data']:
            self.config['data']['file_paths'] = []
        
        self.config['data']['file_paths'].append(file_path)
    
    def get_optimization_params(self):
        """
        Get the optimization parameters.
        
        Returns:
        --------
        dict
            Dictionary of optimization parameters
        """
        return {
            'num_dimensionless_groups': self.config['analysis']['num_dimensionless_groups'],
            'binning_bins': self.config['optimization']['binning_bins'],
            'k_nn': self.config['optimization']['k_nn'],
            'cma_options': self.config['optimization']['cma_options'],
            'n_procs': self.config['optimization']['n_procs'],
            'initial_conditions': self.config['optimization']['initial_conditions'],
        } 

    def save_results(self, results):
        """
        Save the results to a file.
        
        Parameters:
        -----------
        results : dict
            Results from the dimensional analysis
        """
        if not self.config['output']['save_results']:
            return
        
        results_dir = self.config['output']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a timestamp-based filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f"results_{timestamp}.pkl")
        
        # Save the results along with the configuration
        with open(output_file, 'wb') as f:
            pkl.dump({
                'config': self.config,
                'results': results
            }, f)
        
        print(f"Results saved to: {output_file}")
        
        # Also save a human-readable summary
        summary_file = os.path.join(results_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("Dimensional Analysis Results\n")
            f.write("===========================\n\n")
            f.write(f"Optimized MI: {results['optimized_MI']}\n\n")
            f.write("Dimensionless Groups:\n")
            for i, label in enumerate(results['dimensionless_labels']):
                f.write(f"Group {i+1}: {label}\n")
            
            f.write("\nDimensionless Coefficients:\n")
            for i, coef in enumerate(results['dimensionless_coefficients']):
                f.write(f"Group {i+1}: {coef}\n")
        
        print(f"Summary saved to: {summary_file}")
    
    def plot_results(self, results, X, Y):
        """
        Plot the results.
        
        Parameters:
        -----------
        results : dict
            Results from the dimensional analysis
        X : numpy.ndarray
            Input data
        Y : numpy.ndarray
            Output data
        """
        if not self.config['output']['plot_results']:
            return
        
        import matplotlib.pyplot as plt
        
        # Create a directory for plots
        results_dir = self.config['output']['results_dir']
        plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot the dimensionless groups against Y
        pi_values = results['pi_values']
        
        # Create timestamp for unique filenames
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Individual plots for each dimensionless group
        for i in range(pi_values.shape[1]):
            plt.figure(figsize=(10, 6))
            plt.scatter(pi_values[:, i], Y, alpha=0.5)
            plt.xlabel(f"Dimensionless Group {i+1}")
            plt.ylabel("Y")
            plt.title(f"Relationship between Y and Dimensionless Group {i+1}")
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            plot_file = os.path.join(plots_dir, f"group_{i+1}_{timestamp}.png")
            plt.savefig(plot_file)
            plt.close()
            
            print(f"Plot saved to: {plot_file}")
        
        # Combined visualization if there are multiple groups
        if pi_values.shape[1] > 1:
            plt.figure(figsize=(10, 6))
            
            if pi_values.shape[1] == 2:
                # 2D scatter plot for two groups
                plt.scatter(pi_values[:, 0], pi_values[:, 1], c=Y.flatten(), cmap='viridis')
                plt.colorbar(label='Y')
                plt.xlabel(f"Dimensionless Group 1")
                plt.ylabel(f"Dimensionless Group 2")
            elif pi_values.shape[1] == 3:
                # 3D scatter plot for three groups
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(pi_values[:, 0], pi_values[:, 1], pi_values[:, 2], 
                                    c=Y.flatten(), cmap='viridis')
                plt.colorbar(scatter, label='Y')
                ax.set_xlabel(f"Dimensionless Group 1")
                ax.set_ylabel(f"Dimensionless Group 2")
                ax.set_zlabel(f"Dimensionless Group 3")
            
            plt.title("Relationships Between Dimensionless Groups")
            plt.grid(True)
            plt.tight_layout()
            
            # Save the combined plot
            plot_file = os.path.join(plots_dir, f"combined_groups_{timestamp}.png")
            plt.savefig(plot_file)
            plt.close()
            
            print(f"Combined plot saved to: {plot_file}")


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Dimensional Analysis Input System')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--data', '-d', type=str, nargs='+', help='Path(s) to data file(s). Multiple files can be provided.')
    parser.add_argument('--output', '-o', type=str, help='Directory to save results')
    parser.add_argument('--generate-config', '-g', type=str, help='Generate a default configuration file')
    parser.add_argument('--random-sampling', '-r', action='store_true',
                      help='Use random sampling when limiting sample size')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Example usage with command line interface
    args = parse_args()
    
    if args.generate_config:
        # Generate a default configuration file
        input_system = DimensionalAnalysisInput()
        input_system.save_config(args.generate_config)
        print(f"Default configuration saved to: {args.generate_config}")
        sys.exit(0)
    
    # Initialize the input system
    input_system = DimensionalAnalysisInput(args.config)
    
    # Override config with command line arguments
    if args.data:
        input_system.config['data']['file_paths'] = args.data
    if args.output:
        input_system.config['output']['results_dir'] = args.output
    if args.merge:
        input_system.config['data']['merge_strategy'] = args.merge
    if args.random_sampling:
        input_system.config['data']['random_sampling'] = True
    
    # Load data
    X, Y, variable_names, dimension_matrix, basis_matrices = input_system.load_data()
    
    # Get optimization parameters
    opt_params = input_system.get_optimization_params()
    
    # Run dimensional analysis
    results = dimensional_analysis(
        X, 
        Y, 
        dimension_matrix, 
        variable_names, 
        opt_params['num_dimensionless_groups'],
        basis_matrices=basis_matrices,
        cma_options=opt_params['cma_options'],
        binning_bins=opt_params['binning_bins'],
        k_nn=opt_params['k_nn'],
        n_procs=opt_params['n_procs']
    )
    
    # Save results
    input_system.save_results(results)
    
    # Plot results
    input_system.plot_results(results, X, Y)
    
    # Print summary
    print("Optimized MI:", results["optimized_MI"])
    print("\nDimensionless Groups:")
    for i, label in enumerate(results["dimensionless_labels"]):
        print(f"Group {i+1}: {label}")

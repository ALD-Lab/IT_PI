#!/usr/bin/env python
"""
Utility script for converting between pickle and CSV file formats 
for the Dimensional Analysis Tool.


Examples:

1. Convert a single pickle file to CSV format:
! NOTE: Hedaers are optional for CSV files. If not provided, numeric indices will be used.
!       It also needs to match the number of columns in the data.
!
python convert_pkl_and_csv.py --mode to_csv --input data/file.pkl --output data/file.csv --data-key unnormalized_inputs --headers y U_1 nu utau up U_2 U_3 U_4

2. Convert a single CSV file to pickle format:
python convert_pkl_and_csv.py --mode to_pickle --input data/file.csv --output data/file.pkl

3. Convert all pickle files in a directory to CSV format:
python convert_pkl_and_csv.py --mode batch --input-dir data/csv --output-dir data/pickle

4. Convert all pickle files in a directory to CSV format with verbose output:
python convert_pkl_and_csv.py --mode batch --input-dir data/pickle --output-dir data/csv --recursive --verbose

"""

import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
from pathlib import Path


def convert_pickle_to_csv(pickle_path, csv_path, data_key, headers=None, verbose=False):
    """
    Convert a pickle file to CSV format using the specified data key.
    
    Parameters:
    -----------
    pickle_path : str
        Path to the input pickle file
    csv_path : str
        Path where the CSV file will be saved
    data_key : str
        Key to extract data from the pickle file
    headers : list, optional
        List of column names to use as headers. If None, numeric indices will be used.
    verbose : bool, optional
        Whether to print additional information
        
    Returns:
    --------
    bool
        True if conversion successful, False otherwise
    """
    try:
        if verbose:
            print(f"Loading pickle file: {pickle_path}")
        
        # Load the pickle file
        with open(pickle_path, 'rb') as f:
            data = pkl.load(f)
        
        # Extract the data using the specified key
        if data_key not in data:
            raise KeyError(f"Data key '{data_key}' not found in the data file: {pickle_path}")
        
        data_unnormalized = data[data_key]
        
        if verbose:
            print(f"Found data with shape: {data_unnormalized.shape}")
            print(f"Available keys in pickle: {list(data.keys())}")
        
        # Convert to pandas DataFrame with appropriate headers
        if headers:
            # Check if headers length matches column count
            if len(headers) != data_unnormalized.shape[1]:
                print(f"Warning: Number of headers ({len(headers)}) doesn't match number of columns ({data_unnormalized.shape[1]})")
                # Use provided headers up to the column count, or pad with indices if needed
                if len(headers) < data_unnormalized.shape[1]:
                    headers = headers + [f"col_{i}" for i in range(len(headers), data_unnormalized.shape[1])]
                else:
                    headers = headers[:data_unnormalized.shape[1]]
            
            df = pd.DataFrame(data_unnormalized, columns=headers)
        else:
            # Use default column names (col_0, col_1, etc.)
            df = pd.DataFrame(data_unnormalized, columns=[f"col_{i}" for i in range(data_unnormalized.shape[1])])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        
        # Save to CSV with headers
        df.to_csv(csv_path, index=False)
        
        print(f"Successfully converted {pickle_path} to {csv_path}")
        
        if verbose:
            print(f"CSV file shape: {df.shape}")
            print(f"CSV file columns: {df.columns.tolist()}")
        
        return True
        
    except (FileNotFoundError, IOError, KeyError) as e:
        print(f"Error converting file: {e}")
        return False


def convert_csv_to_pickle(csv_path, pickle_path, data_key="unnormalized_inputs", has_header=True, verbose=False):
    """
    Convert a CSV file to pickle format.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    pickle_path : str
        Path where the pickle file will be saved
    data_key : str
        Key to use for storing the data in the pickle file
    has_header : bool
        Whether the CSV file has a header row
    verbose : bool
        Whether to print additional information
        
    Returns:
    --------
    bool
        True if conversion successful, False otherwise
    """
    try:
        if verbose:
            print(f"Loading CSV file: {csv_path}")
        
        # Load the CSV file
        df = pd.read_csv(csv_path, header=0 if has_header else None)
        
        if verbose:
            print(f"Loaded CSV with shape: {df.shape}")
            if has_header:
                print(f"CSV columns: {df.columns.tolist()}")
        
        # Convert to numpy array
        data_array = df.values
        
        # Create a dictionary with the data
        data_dict = {
            data_key: data_array,
            'csv_columns': df.columns.tolist() if has_header else None,
            'csv_source': csv_path,
            'conversion_info': {
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'original_shape': data_array.shape
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(pickle_path)), exist_ok=True)
        
        # Save to pickle
        with open(pickle_path, 'wb') as f:
            pkl.dump(data_dict, f)
        
        print(f"Successfully converted {csv_path} to {pickle_path}")
        
        if verbose:
            print(f"Pickle data stored under key: '{data_key}'")
            print(f"Additional keys in pickle: {list(data_dict.keys())}")
        
        return True
        
    except (FileNotFoundError, IOError, pd.errors.EmptyDataError) as e:
        print(f"Error converting file: {e}")
        return False


def batch_convert(input_dir, output_dir, mode, data_key, recursive=False, verbose=False):
    """
    Batch convert files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing files to convert
    output_dir : str
        Directory where converted files will be saved
    mode : str
        'to_csv' or 'to_pickle'
    data_key : str
        Key for accessing/storing data in pickle files
    recursive : bool
        Whether to search for files recursively
    verbose : bool
        Whether to print additional information
        
    Returns:
    --------
    int
        Number of successfully converted files
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files
    input_path = Path(input_dir)
    if recursive:
        if mode == 'to_csv':
            files = list(input_path.rglob("*.pkl"))
        else:
            files = list(input_path.rglob("*.csv"))
    else:
        if mode == 'to_csv':
            files = list(input_path.glob("*.pkl"))
        else:
            files = list(input_path.glob("*.csv"))
    
    if verbose:
        print(f"Found {len(files)} files to convert")
    
    success_count = 0
    for file_path in files:
        # Get relative path for preserving directory structure
        rel_path = file_path.relative_to(input_dir) if recursive else file_path.name
        
        if mode == 'to_csv':
            # Convert .pkl to .csv
            output_path = os.path.join(output_dir, str(rel_path).replace('.pkl', '.csv'))
            success = convert_pickle_to_csv(str(file_path), output_path, data_key, verbose=verbose)
        else:
            # Convert .csv to .pkl
            output_path = os.path.join(output_dir, str(rel_path).replace('.csv', '.pkl'))
            success = convert_csv_to_pickle(str(file_path), output_path, data_key, verbose=verbose)
        
        if success:
            success_count += 1
    
    print(f"Successfully converted {success_count} out of {len(files)} files")
    return success_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert between pickle and CSV formats for Dimensional Analysis')
    
    # Main operation mode
    parser.add_argument('--mode', choices=['to_csv', 'to_pickle', 'batch'], required=True,
                        help='Conversion mode: to_csv, to_pickle, or batch')
    
    # Single file conversion arguments
    parser.add_argument('--input', type=str, help='Input file (for single file conversion)')
    parser.add_argument('--output', type=str, help='Output file (for single file conversion)')
    
    # Batch conversion arguments
    parser.add_argument('--input-dir', type=str, help='Input directory (for batch conversion)')
    parser.add_argument('--output-dir', type=str, help='Output directory (for batch conversion)')
    parser.add_argument('--recursive', action='store_true', help='Search for files recursively in batch mode')
    
    # Common parameters
    parser.add_argument('--data-key', type=str, default='unnormalized_inputs',
                        help='Key for accessing/storing data in pickle files')
    parser.add_argument('--headers', type=str, nargs='+', help='Headers for CSV file (for to_csv mode)')
    parser.add_argument('--no-header', action='store_true', help='CSV has no header (for to_pickle mode)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print additional information')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    try:
        if args.mode == 'to_csv':
            # Convert pickle to CSV
            if not args.input or not args.output:
                print("Error: --input and --output are required for single file conversion")
                return 1
            
            success = convert_pickle_to_csv(
                args.input,
                args.output,
                args.data_key,
                args.headers,
                args.verbose
            )
            return 0 if success else 1
            
        elif args.mode == 'to_pickle':
            # Convert CSV to pickle
            if not args.input or not args.output:
                print("Error: --input and --output are required for single file conversion")
                return 1
                
            success = convert_csv_to_pickle(
                args.input,
                args.output,
                args.data_key,
                not args.no_header,
                args.verbose
            )
            return 0 if success else 1
            
        elif args.mode == 'batch':
            # Batch conversion
            if not args.input_dir or not args.output_dir:
                print("Error: --input-dir and --output-dir are required for batch conversion")
                return 1
                
            success_count = batch_convert(
                args.input_dir,
                args.output_dir,
                'to_csv' if args.mode == 'batch' and args.input_dir.endswith('pkl') else 'to_pickle',
                args.data_key,
                args.recursive,
                args.verbose
            )
            return 0 if success_count > 0 else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

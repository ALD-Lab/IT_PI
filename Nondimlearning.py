'''
Utilities used for nondimensional learning

WARNING: Only bin-based mutual information is supported for now.
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank, inv
from scipy.special import erf, psi
import scipy.spatial as scispa
import warnings
import random
from cma import CMAEvolutionStrategy

# NOTE: For multiprocessing
from cma.optimization_tools import EvalParallel2
from functools import partial

def dimensional_analysis(X, Y, D_in, variable_names, num_input, 
                         basis_matrices=None,
                         cma_options=None, 
                         n_procs=20,
                         binning_bins=50, 
                         initial_conditions=None,
                         k_nn=5):
    """
    Perform dimensional analysis to find dimensionless groups.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input array of dimensional parameters (n_samples, n_variables)
    Y : numpy.ndarray
        Output array to be non-dimensionalized (n_samples, 1)
    D_in : numpy.ndarray
        Dimension matrix describing the dimensions of each input variable
    variable_names : list
        List of variable names corresponding to columns in X
    num_input : int
        Number of dimensionless groups to find
    cma_options : dict, optional
        Options for CMA-ES optimizer
    basis_matrices : numpy.ndarray
        Option to provide customzied basis matrix
    n_procs: int, optional
        Number of processors for multiprocessing
    binning_bins : int, optional
        Number of bins for MI calculation
    k_nn : int, optional
        K value for nearest-neighbor in Kraskov MI calculation
        
    Returns:
    --------
    dict
        Dictionary containing dimensionless groups and related information
    """
    # NOTE: Ensure inputs are in the right format
    if not isinstance(Y, np.ndarray) or Y.ndim != 2:
        Y = np.array(Y).reshape(-1, 1)
    
    # Ensure D_in is a numpy array
    D_in = np.array(D_in)
    
    # NOTE: Calculate basis matrices
    rank_D_in = matrix_rank(D_in)
    num_basis = D_in.shape[1] - rank_D_in
    # NOTE: Provide option to use custom basis matrices
    if basis_matrices is None:
        basis_matrices = calc_basis(D_in, num_basis)
    print("Basis matrices:\n", basis_matrices)
    
    # NOTE: Set the number of parameters to perform optimization
    num_params = num_basis * num_input
    
    # NOTE: Set default CMA options if not provided
    if cma_options is None:
        lower_bounds = [-2] * num_params
        upper_bounds = [2] * num_params
        bounds = [lower_bounds, upper_bounds]
        cma_options = {
            'bounds': bounds,
            'maxiter': 50000,
            'tolx': 1e-4,
            'tolfun': 1e-4,
            'popsize': 300,
        }
    
    # NOTE: Run CMA optimization
    
    # NOTE: 1. Set initial conditions
    if initial_conditions is None:
        # Default initial conditions (all set to 0.1) based on number of parameters
        initial_solution = [0.1] * num_params
    else:
        # Use provided initial conditions
        if len(initial_conditions) != num_params:
            warnings.warn(f"Expected {num_params} initial conditions but got {len(initial_conditions)}. Using default.")
            initial_solution = [0.1] * num_params
        else:
            initial_solution = initial_conditions
    print(f"Using initial conditions: {initial_solution}")
            
    # NOTE: 2. Set up CMA-ES optimizer
    es = CMAEvolutionStrategy(initial_solution, 0.5, cma_options)
    initial_MI = MI_input_output(initial_solution, basis_matrices, X, Y, num_basis, num_input, binning_bins)
    print("Initial MI:", initial_MI)
    
    # NOTE: 3. Run optimization

    # NOTE: 3.1 Use serial evaluation if n_procs is set to 1
    if n_procs == 1:
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [MI_input_output(x, basis_matrices, X, Y, num_basis, num_input, binning_bins) for x in solutions])
            es.disp()
    else:
    # NOTE: 3.2 Use multiprocessing if n_procs is greater than 1
        evaluate = partial(
            MI_input_output, 
            basis_matrices=basis_matrices,
            X=X, 
            Y=Y, 
            num_basis=num_basis, 
            num_inputs=num_input, 
            num_bins=binning_bins
        )

        with EvalParallel2(evaluate, n_procs) as eval_all:
            while not es.stop():
                sol = es.ask()
                es.tell(sol, eval_all(sol))  
                es.disp()

    # NOTE: 4 Return results
    
    # NOTE: 4.1 Extract optimized parameters
    optimized_params = es.result.xbest
    optimized_MI = es.result.fbest
    
    # NOTE: 4.2 Create dimensionless groups
    a_list = [tuple(optimized_params[i*num_basis:(i+1)*num_basis]) for i in range(num_input)]
    # Use np.array to ensure correct shape for matrix multiplication
    coef_pi_list = []
    for a in a_list:
        a_array = np.array(a).reshape(1, -1)
        coef = np.matmul(a_array, basis_matrices)
        coef_pi_list.append(coef.flatten())
    
    # NOTE: 4.3 Normalize coefficients
    normalized_coef_pi_list = []
    for coef_pi in coef_pi_list:
        max_abs_value = np.max(np.abs(coef_pi))
        if max_abs_value > 0:
            normalized_coef_pi = coef_pi / max_abs_value
            normalized_coef_pi_list.append(np.round(normalized_coef_pi, 1))
        else:
            normalized_coef_pi_list.append(coef_pi)
    
    # NOTE: 4.4 Create LaTeX labels for dimensionless groups
    omega_groups = np.array(normalized_coef_pi_list).reshape(-1, len(variable_names))
    labels = create_labels(omega_groups, variable_names)
    
    # NOTE: 4.5 Create dimensionless variables for validation
    pi_list = [calc_pi(a, basis_matrices, X) for a in a_list]
    pi_array = np.column_stack(pi_list)
    
    return {
        "optimized_params": optimized_params,
        "optimized_MI": optimized_MI,
        "a_list": a_list,
        "dimensionless_coefficients": normalized_coef_pi_list,
        "dimensionless_labels": labels,
        "pi_values": pi_array,
        "basis_matrices": basis_matrices
    }


def calc_basis(D_in, col_range):
    """
    Calculate basis matrices for the dimensional analysis.
    
    Parameters:
    -----------
    D_in : numpy.ndarray
        Dimension matrix
    col_range : int
        Range for the column calculations
        
    Returns:
    --------
    numpy.ndarray
        Basis matrices
    """
    num_rows = np.shape(D_in)[0]
    Din1, Din2 = D_in[:, :num_rows], D_in[:, num_rows:]
    basis_matrices = []
    
    for i in range(col_range):
        x2 = np.zeros((col_range, 1))
        x2[i, 0] = -1
        # Modified matrix multiplication to work with ndarrays
        x1 = -np.matmul(inv(Din1), np.matmul(Din2, x2))
        basis_matrices.append(np.vstack((x1, x2)))
    
    # Ensure proper shape for basis matrices (2D for proper dot product)
    return np.array([m.reshape(-1) for m in basis_matrices])


def calc_pi(c, basis_matrices, X, threshold=0.1):
    """
    Calculate Pi values based on coefficients and basis matrices.
    
    Parameters:
    -----------
    c : tuple
        Coefficients of each dimensionless group
    basis_matrices : numpy.ndarray
        Basis matrices
    X : numpy.ndarray
        Input data
    threshold : float
        threshold for which the sign of the input variables is considered 

    Returns:
    --------
    numpy.ndarray
        Pi values
    """
    # Ensure c is properly shaped for matrix multiplication
    c_array = np.array(c).reshape(1, -1) if isinstance(c, tuple) else np.array(c).reshape(1, -1)
    
    # Adjust dimensions for proper matrix multiplication
    # If basis_matrices has shape (3,5,1), reshape it to (3,5)
    if basis_matrices.ndim == 3:
        basis_matrices_2d = basis_matrices.reshape(basis_matrices.shape[0], basis_matrices.shape[1])
    else:
        basis_matrices_2d = basis_matrices
        
    coef_pi = np.matmul(c_array, basis_matrices_2d).flatten()
    pi_mat = np.ones((X.shape[0], 1))

    # WARNING: Find the product of all signs of the input variables
    sign_mat_ = np.sign(X)
    
    for i in range(len(coef_pi)):
        if i < X.shape[1]:  # Make sure we don't go out of bounds
            tmp = np.abs(X[:, i]) ** coef_pi[i]

            # NOTE: Give signs to the nondimensional groups; 
            # The sign of the input variables is considered if the coefficient is greater than the threshold
            
            if abs(coef_pi[i]) > threshold:
                sign_mat_i = sign_mat_[:, i]
                tmp = np.multiply(tmp, sign_mat_i)
            pi_mat = np.multiply(pi_mat, tmp.reshape(-1, 1))
    
    return pi_mat


def MI_d_binning(input_data, output_data, num_bins):
    """
    Calculate mutual information using binning.
    
    Parameters:
    -----------
    input_data : numpy.ndarray
        Input data
    output_data : numpy.ndarray
        Output data
    num_bins : int
        Number of bins
        
    Returns:
    --------
    float
        Mutual information value
    """
    def entropy_bin(X, num_bins):
        N, D = X.shape
        bins = [num_bins] * D
        hist, _ = np.histogramdd(X, bins=bins)
        hist = hist / np.sum(hist)  # Normalize to get probabilities
        positive_indices = hist > 0
        return -np.sum(hist[positive_indices] * np.log(hist[positive_indices]))

    mi = entropy_bin(input_data, num_bins) + entropy_bin(output_data, num_bins) - entropy_bin(np.hstack([input_data, output_data]), num_bins)
    return mi


def KraskovMI1_nats(x, y, k=1):
    """
    Compute mutual information using the Kraskov estimator.
    
    Parameters:
    -----------
    x : numpy.ndarray
        First variable
    y : numpy.ndarray
        Second variable
    k : int
        Nearest-neighbor parameter
        
    Returns:
    --------
    float
        Mutual information value
    """
    N, dim = x.shape
    V = np.hstack([x, y])

    # Init query tree
    kdtree = scispa.KDTree(V)
    ei, _ = kdtree.query(V, k + 1, p=np.infty)
    # infty norm is gonna give us the maximum distance
    dM = ei[:, -1]

    kdtree_x = scispa.KDTree(x)
    kdtree_y = scispa.KDTree(y)

    nx = kdtree_x.query_ball_point(x, dM, p=np.infty, return_length=True)
    ny = kdtree_y.query_ball_point(y, dM, p=np.infty, return_length=True)

    # we do not add + 1 because it is accounted in query_ball_point
    ave = (psi(nx) + psi(ny)).mean()

    return psi(k) - ave + psi(N)


def MI_input_output(para, basis_matrices, X, Y, num_basis, num_inputs, num_bins=50):
    """
    Calculate mutual information between input and output.
    
    Parameters:
    -----------
    para : list
        Parameters for evaluation
    basis_matrices : numpy.ndarray
        Basis matrices
    X : numpy.ndarray
        Input data
    Y : numpy.ndarray
        Output data
    num_basis : int
        Number of basis matrices
    num_inputs : int
        Number of inputs
    num_bins : int
        Number of bins for binning method
        
    Returns:
    --------
    float
        Negative mutual information (for minimization)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        a_list = [tuple(para[i*num_basis:(i+1)*num_basis]) for i in range(num_inputs)]
        try:
            # Use try-except to catch any matrix dimension errors
            pi_list = []
            for a in a_list:
                try:
                    pi = calc_pi(a, basis_matrices, X)
                    pi_list.append(pi)
                except (ValueError, RuntimeWarning) as e:
                    # If there's an error, return a large positive value
                    print(f"Error in calc_pi: {str(e)}")
                    return random.uniform(1e6, 1e10)
                    
            pi = np.column_stack(pi_list)
        except (RuntimeWarning, ValueError) as e:
            print(f"Error in MI_input_output: {str(e)}")
            return random.uniform(1e6, 1e10)  # Return a large positive value in case of exceptions
    
    if np.any(np.isnan(pi)):
        return random.uniform(1e6, 1e10)  # Return a large positive value
    
    Y = Y.reshape(-1, 1)
    MI = MI_d_binning(np.array(pi), np.array(Y), num_bins)
    # Alternatively: MI = KraskovMI1_nats(np.array(pi), np.array(Y), 5)
    return -MI


def create_labels(omega, variables):
    """
    Create LaTeX labels for the dimensionless groups.
    
    Parameters:
    -----------
    omega : numpy.ndarray
        Array of exponents for each variable
    variables : list
        List of variable names
        
    Returns:
    --------
    list
        List of LaTeX formatted labels
    """
    labels = []
    for row in omega:
        positive_terms = []
        negative_terms = []
        
        for i, value in enumerate(row):
            if abs(value) < 1e-10:  # Handle zero values
                continue
                
            var_name = variables[i]
            
            if value > 0:
                if value == 1:
                    positive_terms.append(f"{var_name}")
                else:
                    # Format the exponent with fixed precision to avoid floating point issues
                    exponent = f"{value:.1f}".rstrip('0').rstrip('.') if value != int(value) else str(int(value))
                    positive_terms.append(f"{var_name}^{{{exponent}}}")
            elif value < 0:
                abs_value = abs(value)
                if abs_value == 1:
                    negative_terms.append(f"{var_name}")
                else:
                    # Format the exponent with fixed precision
                    exponent = f"{abs_value:.1f}".rstrip('0').rstrip('.') if abs_value != int(abs_value) else str(int(abs_value))
                    negative_terms.append(f"{var_name}^{{{exponent}}}")
        
        # Join terms with multiplication
        numerator = " \\cdot ".join(positive_terms) if positive_terms else "1"
        denominator = " \\cdot ".join(negative_terms) if negative_terms else "1"
        
        # Create the final label
        if denominator == "1":
            labels.append(f"${numerator}$")
        elif numerator == "1":
            labels.append(f"$\\frac{{1}}{{" + denominator + "}}$")
        else:
            labels.append(f"$\\frac{{{numerator}}}{{{denominator}}}$")
    
    return labels

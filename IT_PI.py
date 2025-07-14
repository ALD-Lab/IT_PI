import numpy as np
from scipy.special import erf, psi
from numpy.linalg import inv, matrix_rank
import scipy.spatial as scispa
import warnings
import random
from cma import CMAEvolutionStrategy
from pprint import pprint


###############################################
# Basis computation (Null space of D_in)
###############################################
def calc_basis(D_in, col_range):
    num_rows = np.shape(D_in)[0]
    Din1, Din2 = D_in[:, :num_rows], D_in[:, num_rows:]
    basis_matrices = []
    for i in range(col_range):
        x2 = np.zeros((col_range, 1))
        x2[i, 0] = -1
        x1 = -inv(Din1) * Din2 * x2
        basis_matrices.append(np.vstack((x1, x2)))
    return np.asmatrix(np.array(basis_matrices))

###############################################
# Construct dimensionless variables
###############################################
def calc_pi(c, basis_matrices, X):
    coef_pi = np.dot(c, basis_matrices)
    pi_mat = np.ones((X.shape[0], 1))
    for i in range(coef_pi.shape[1]):
        tmp = X[:, i] ** coef_pi[:, i]
        pi_mat = np.multiply(pi_mat, tmp.reshape(-1, 1))
    return pi_mat

def calc_pi_omega(coef_pi, X):
    pi_mat = np.ones((X.shape[0], 1))
    for i in range(coef_pi.shape[1]):
        tmp = X[:, i] ** coef_pi[:, i]
        pi_mat = np.multiply(pi_mat, tmp.reshape(-1, 1))
    return pi_mat

###############################################
# Mutual information estimators
###############################################
def MI_d_binning(input, output, num_bins):
    def entropy_bin(X, num_bins):
        N, D = X.shape
        bins = [num_bins] * D
        hist, _ = np.histogramdd(X, bins=bins)
        hist = hist / np.sum(hist)
        positive_indices = hist > 0
        return -np.sum(hist[positive_indices] * np.log(hist[positive_indices]))
    mi = entropy_bin(input, num_bins) + entropy_bin(output, num_bins) - entropy_bin(np.hstack([input, output]), num_bins)
    return mi

def KraskovMI1_nats(x, y, k=1):
    N, dim = x.shape
    V = np.hstack([x, y])
    kdtree = scispa.KDTree(V)
    ei, _ = kdtree.query(V, k+1, p=np.inf)
    dM = ei[:, -1]
    nx = scispa.KDTree(x).query_ball_point(x, dM, p=np.inf, return_length=True)
    ny = scispa.KDTree(y).query_ball_point(y, dM, p=np.inf, return_length=True)
    ave = (psi(nx) + psi(ny)).mean()
    return psi(k) - ave + psi(N)

###############################################
# Objective function for CMA-ES
###############################################
def MI_input_output(
    para,
    basis_matrices,
    X,
    Y,
    num_basis,
    num_inputs,
    estimator="binning",      # "binning" or "kraskov"
    estimator_params=None     # dict of hyperparameters
):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        a_list = [tuple(para[i*num_basis:(i+1)*num_basis]) for i in range(num_inputs)]
        try:
            pi_list = [calc_pi(a, basis_matrices, X) for a in a_list]
            pi = np.column_stack(pi_list)
        except RuntimeWarning:
            return random.uniform(1e6, 1e10)

    if np.any(np.isnan(pi)):
        return random.uniform(1e6, 1e10)

    Y = Y.reshape(-1, 1)

    # Choose estimator
    if estimator == "binning":
        num_bins = estimator_params.get("num_bins", 50) if estimator_params else 50
        mi = MI_d_binning(np.array(pi), np.array(Y), num_bins)

    elif estimator == "kraskov":
        k = estimator_params.get("k", 5) if estimator_params else 5
        mi = KraskovMI1_nats(np.array(pi), np.array(Y), k)

    else:
        raise ValueError(f"Unknown estimator '{estimator}'. Use 'binning' or 'kraskov'.")

    return -mi


###############################################
# Labels generation
###############################################
def create_labels(omega, variables):
    labels = []
    for row in omega:
        positive_part = ''
        negative_part = ''
        for i, value in enumerate(row):
            value = float(value)  # Safe scalar cast
            if value > 0:
                if positive_part == '':
                    positive_part = f"{variables[i]}^{{{value}}}"
                else:
                    positive_part += f" \\cdot {variables[i]}^{{{value}}}"
            elif value < 0:
                if negative_part == '':
                    negative_part = f"{variables[i]}^{{{-value}}}"
                else:
                    negative_part += f" \\cdot {variables[i]}^{{{-value}}}"
        if negative_part == '':
            labels.append(f"${positive_part}$")
        elif positive_part == '':
            labels.append(f"$\\frac{{1}}{{{negative_part}}}$")
        else:
            labels.append(f"$\\frac{{{positive_part}}}{{{negative_part}}}$")
    return labels

###############################################
# Compute irreducible error and uncertainty
###############################################
def calculate_bound_and_uq(input_data, output_data, num_trials):
    mi_full = KraskovMI1_nats(input_data, output_data, 5)
    epsilon_full = np.exp(-mi_full)
    epsilon_half_values = []
    for _ in range(num_trials):
        idx = np.random.choice(input_data.shape[0], input_data.shape[0]//2, replace=False)
        mi_half = KraskovMI1_nats(input_data[idx], output_data[idx], 5)
        epsilon_half_values.append(np.exp(-mi_half))
    epsilon_half_mean = np.mean(epsilon_half_values)
    uq = abs(epsilon_full - epsilon_half_mean)
    return epsilon_full, uq

###############################################
# Main function
# X: Input data matrix 
# Y: Output data vector 
# D_in: Dimension matrix for input variables
# variables: List of variable names for dimensionless variables
# num_input: Number of input variables (default is 1)
# popsize: Population size for CMA-ES (default is 300)
# maxiter: Maximum number of iterations for CMA-ES (default is 50000)
# num_trials: Number of trials for uncertainty calculation (default is 10)
# estimator: Mutual information estimator to use ('binning' or 'kraskov', default is 'binning')
# estimator_params: Dictionary of hyperparameters for the estimator (default is None, which uses default values)    
# Returns labels for optimal dimensionless variables, irreducible error, and uncertainty.
###############################################
def main(
    X,
    Y,
    D_in,
    variables,
    num_input=1,
    popsize=300,
    maxiter=50000,
    num_trials=10,
    estimator="binning",
    estimator_params=None,
    seed = None
):
    print("Rank of D_in:", matrix_rank(D_in))
    print("D_in matrix:\n", D_in)
    # Calculate the basis matrices
    num_basis = D_in.shape[1] - matrix_rank(D_in)
    basis_matrices = calc_basis(D_in, num_basis)
    print("Basis vectors:")
    pprint(basis_matrices)
    
    print('-'*60)
    # Hom many coeeficients need to be optimized?
    num_para = num_basis * num_input
    print('num of parameters:', num_para)
    
    # CMA-ES setup
    lower_bounds = [-2] * num_para
    upper_bounds = [2] * num_para
    bounds = [lower_bounds, upper_bounds]
    options = {
        'bounds': bounds,
        'maxiter': maxiter,
        'tolx': 1e-4,
        'tolfun': 1e-4,
        'popsize': popsize,
        'seed': seed if seed is not None else random.randint(0, 10000),
    }
    
    # Mutual information estimator setup
    if estimator_params is None:
        if estimator == "binning":
            estimator_params = {"num_bins": 50}
        elif estimator == "kraskov":
            estimator_params = {"k": 5}
    print(f"\nUsing estimator: '{estimator}' with hyperparameters: {estimator_params}\n")

   # Run CMA-ES
    es = CMAEvolutionStrategy([0.1]*num_para, 0.5, options)
    while not es.stop():
        solutions = es.ask()
        es.tell(
            solutions,
            [
                MI_input_output(
                    x,
                    basis_matrices,
                    X,
                    Y,
                    num_basis,
                    num_input,
                    estimator=estimator,
                    estimator_params=estimator_params
                )
                for x in solutions
            ]
        )
        es.disp()
    es.result_pretty()
    optimized_params = es.result.xbest
    optimized_MI = es.result.fbest
    print('Optimized_params:', optimized_params)
    print('Optimized_MI:', optimized_MI)
    print('-'*60)

    # Normalize the dimensionless variables coefficients
    a_list = [tuple(optimized_params[i*num_basis:(i+1)*num_basis]) for i in range(num_input)]
    # print('a_list:', a_list)
    coef_pi_list = [np.dot(a, basis_matrices) for a in a_list]
    normalized_coef_pi_list = []
    for coef_pi in coef_pi_list:
        max_abs_value = np.max(np.abs(coef_pi))
        normalized_coef_pi = coef_pi / max_abs_value
        normalized_coef_pi_list.append(np.round(normalized_coef_pi, 2))
        print('Coef_pi:', normalized_coef_pi)
    # Create labels for the dimensionless variables
    omega = np.array(normalized_coef_pi_list).reshape(-1, len(variables))
    optimal_pi_lab = create_labels(omega, variables)
    for j, label in enumerate(optimal_pi_lab):
        print(f'Optimal_pi_lab[{j}] = {label}')
        
    # Calculate the dimensionless variables data
    input_list = [calc_pi_omega(np.array(omega), X) for omega in normalized_coef_pi_list]
    input_PI = np.column_stack(input_list)
    
    # Calculate irreducible error and uncertainty
    epsilon_values = []
    uq_values = []
    x_labels = []
    for j in range(input_PI.shape[1]):
        epsilon_full, uq = calculate_bound_and_uq(
            input_PI[:, j].reshape(-1, 1), Y, num_trials
        )
        epsilon_values.append(epsilon_full)
        uq_values.append(uq)
        x_labels.append(f"$\\Pi_{j+1}^*$")

    if input_PI.shape[1] > 1:
        epsilon_full, uq = calculate_bound_and_uq(input_PI, Y, num_trials)
        epsilon_values.append(epsilon_full)
        uq_values.append(uq)
        label = "$[" + ",".join([f"\\Pi_{i+1}^*" for i in range(input_PI.shape[1])]) + "]$"
        x_labels.append(label)

    return {
        "optimized_params": optimized_params,
        "labels": optimal_pi_lab,
        "irreducible_error": epsilon_values,
        "uncertainty": uq_values
    }

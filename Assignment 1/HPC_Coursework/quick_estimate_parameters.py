#!/usr/bin/env python

import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')  # Prevents the need for X server on HPC
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import minimize

def save_data_for_plotting(data, filename):
    """Saves a Python object to file with pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def reshape_params_to_ab(params, K):
    """
    Given the 1D array of log(a_{k,j}), log(b_k),
    reconstruct and exponentiate to get a_{k,j} and b_k.
    """
    num_a = K*(K-1)
    log_a_flat = params[:num_a]
    log_b = params[num_a:]
    
    # Build a, skipping diagonal
    a = np.zeros((K, K))
    idx = 0
    for k in range(K):
        for j in range(K):
            if j != k:
                a[k, j] = np.exp(log_a_flat[idx])
                idx += 1
    b = np.exp(log_b)
    return a, b

def conditional_log_likelihood(params, X):
    """
    Negative conditional log-likelihood and gradient
    for p(x_k | x_{-k}) = Normal(0, sum_{j != k} a_{k,j} x_j^2 + b_k).
    """
    N, K = X.shape

    # Reshape log_a -> a and log_b -> b
    a, b = reshape_params_to_ab(params, K)

    log_likelihood = 0.0
    grad_log_a = np.zeros_like(a)   # shape (K, K)
    grad_log_b = np.zeros(K)        # shape (K,)

    for n in range(N):
        x_n = X[n, :]
        for k in range(K):
            sigma_sq = np.sum(a[k, :] * x_n**2) - a[k, k]*(x_n[k]**2) + b[k]
            if not np.isfinite(sigma_sq) or sigma_sq <= 0:
                sigma_sq = 1e-6
            x_nk = x_n[k]
            # Log-likelihood contribution
            log_likelihood += (
                -0.5 * np.log(2*np.pi)
                - 0.5 * np.log(sigma_sq)
                - 0.5 * (x_nk**2)/sigma_sq
            )
            # Gradient wrt a_{k,j}
            for j in range(K):
                if j != k:
                    partial = (
                        -0.5/sigma_sq + 0.5*(x_nk**2)/(sigma_sq**2)
                    ) * a[k, j]*(x_n[j]**2)
                    grad_log_a[k, j] += partial
            # Gradient wrt b_k
            grad_log_b[k] += (
                -0.5/sigma_sq + 0.5*(x_nk**2)/(sigma_sq**2)
            ) * b[k]

    # Convert grad_log_a to log form (like the parameters)
    # We stored a_{k,j} = exp(log_a_{k,j}), so chain rule => multiply by a_{k,j}.
    for k in range(K):
        for j in range(K):
            if j != k:
                grad_log_a[k, j] *= a[k, j]

    # Similarly for b_k
    grad_log_b *= b

    # Flatten grad_log_a (excluding diagonal)
    grad_log_a_flat = []
    for k in range(K):
        for j in range(K):
            if j != k:
                grad_log_a_flat.append(grad_log_a[k, j])
    grad_log_a_flat = np.array(grad_log_a_flat)

    grad = np.concatenate([grad_log_a_flat, grad_log_b])

    return -log_likelihood, -grad  # Negative because we are minimizing

def estimate_parameters(X, max_iter=50, save_dir="./parameter_estimation_checkpoint"):
    """
    Estimate a_{k,j}, b_k by maximizing the conditional log-likelihood.
    Continuously saves partial parameters after every iteration to
    partial_params_last.pkl in case the job stops early.
    """
    os.makedirs(save_dir, exist_ok=True)

    N, K = X.shape
    # Subset data to speed up
    N_train = min(N, 2000)  
    X_train = X[:N_train]

    num_a_params = K*(K-1)
    num_b_params = K
    initial_params = np.random.randn(num_a_params + num_b_params)*0.01

    objective_history = []
    partial_filename = os.path.join(save_dir, "partial_params_last.pkl")

    def callback(params):
        # track iteration as length of objective_history so far
        iteration = len(objective_history)
        obj_val, _ = conditional_log_likelihood(params, X_train)
        objective_history.append(obj_val)

        # Convert to a, b
        a_chk, b_chk = reshape_params_to_ab(params, K)

        # Save partial checkpoint
        partial_data = {
            'iteration': iteration,
            'objective': obj_val,
            'params': params.copy(),
            'a': a_chk,
            'b': b_chk
        }
        save_data_for_plotting(partial_data, partial_filename)

        if iteration % 5 == 0:
            print(f"Iteration {iteration}: Loss = {obj_val:.6f}")

    print("Starting Optimization...")

    # Run the optimizer
    result = minimize(
        lambda p: conditional_log_likelihood(p, X_train),
        initial_params,
        method='L-BFGS-B',
        jac=True,
        callback=callback,
        options={'maxiter': max_iter, 'disp': True}
    )

    # Evaluate final objective
    final_obj, _ = conditional_log_likelihood(result.x, X_train)
    objective_history.append(final_obj)

    if result.success:
        print(f"Optimization successful! Final loss: {result.fun:.6f}")
    else:
        print(f"WARNING: Optimization did not converge. Msg: {result.message}")

    # Convert final solution into a, b
    a_final, b_final = reshape_params_to_ab(result.x, K)

    # Save final results
    params_data = {
        'a': a_final,
        'b': b_final,
        'optimized_params': result.x,
        'objective_history': objective_history
    }
    final_file = os.path.join(save_dir, 'estimated_parameters.pkl')
    save_data_for_plotting(params_data, final_file)

    # Plot quick convergence
    plt.figure()
    plt.plot(objective_history, marker='o', linestyle='-')
    plt.title('Neg. Cond. Log-Likelihood')
    plt.xlabel('Iteration (approx.)')
    plt.ylabel('Objective Value')
    plt.savefig(os.path.join(save_dir, 'convergence.png'))
    plt.close()

    # Plot final a, b
    plt.figure()
    plt.imshow(a_final, cmap='viridis')
    plt.title('Estimated a[k,j]')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'parameter_a_matrix.png'))
    plt.close()

    plt.figure()
    plt.plot(b_final, marker='o')
    plt.title('Estimated b[k]')
    plt.savefig(os.path.join(save_dir, 'parameter_b_values.png'))
    plt.close()

    return a_final, b_final

def main():
    # Adjust to your HPC environment
    mat_file_path = "./representational.mat"
    output_dir = "./representational_output_fast"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from {mat_file_path} ...")
    mat_data = scipy.io.loadmat(mat_file_path)

    # Extract
    Y = mat_data['Y']  # shape (32000, 1024)
    R = mat_data['R']  # shape (1024, 256)

    # Build X
    X = Y @ R  # shape (32000, 256)
    print("Data shape (X):", X.shape)

    # Estimate parameters
    print("\n*** Estimating conditional model parameters w/ checkpoint ***")
    a, b = estimate_parameters(
        X,
        max_iter=50,  # reduce or increase as needed
        save_dir=os.path.join(output_dir, "parameter_estimation_checkpoint")
    )
    print("\nDone! Final a,b saved. Check partial_params_last.pkl for mid-run checkpoints.")

if __name__ == "__main__":
    main()

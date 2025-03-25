#!/usr/bin/env python

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import minimize

def save_data_for_plotting(data, filename):
    """Save data to file for later plotting locally."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def conditional_log_likelihood(params, X):
    """
    Compute the negative conditional log-likelihood and its gradient
    for the model:
        p(x_k | x_{-k}) = Normal(0, sum_j a_{k,j} * x_j^2 + b_k).
    'params' is a 1D array of length (K*(K-1) + K),
    containing log(a_{k,j}) and log(b_k).
    """
    N, K = X.shape

    # Unpack parameters
    log_a_flat = params[:K*(K-1)]
    log_b = params[K*(K-1):]

    # Reshape log_a to a K x K matrix (with zeros on diagonal)
    log_a = np.zeros((K, K))
    idx = 0
    for k in range(K):
        for j in range(K):
            if j != k:
                log_a[k, j] = log_a_flat[idx]
                idx += 1

    # Exponentiate to get a_{k,j} and b_k
    a = np.exp(log_a)
    b = np.exp(log_b)

    # Initialize log-likelihood and gradients
    log_likelihood = 0.0
    grad_log_a = np.zeros_like(log_a)
    grad_log_b = np.zeros_like(log_b)

    # Compute log-likelihood and derivatives
    for n in range(N):
        if n % 1000 == 0:
            print(f"Processing sample {n}/{N}...")
        x_n = X[n, :]
        for k in range(K):
            sigma_squared = np.sum(a[k, :] * x_n**2) - a[k, k] * x_n[k]**2 + b[k]
            if not np.isfinite(sigma_squared) or sigma_squared <= 0:
                # Prevent invalid or zero value
                sigma_squared = 1e-6
            x_nk = x_n[k]
            # Contribution to log-likelihood
            log_likelihood += -0.5 * np.log(2*np.pi) - 0.5 * np.log(sigma_squared) - 0.5*(x_nk**2)/sigma_squared

            # Gradient wrt a_{k,j}
            for j in range(K):
                if j != k:
                    partial = (-0.5 / sigma_squared + 0.5*(x_nk**2)/(sigma_squared**2)) * a[k, j] * x_n[j]**2
                    grad_log_a[k, j] += partial

            # Gradient wrt b_k
            grad_log_b[k] += (-0.5/sigma_squared + 0.5*(x_nk**2)/(sigma_squared**2)) * b[k]

    # Flatten grad_log_a (excluding diagonal)
    grad_log_a_flat = []
    for k in range(K):
        for j in range(K):
            if j != k:
                grad_log_a_flat.append(grad_log_a[k, j])
    grad_log_a_flat = np.array(grad_log_a_flat)

    grad = np.concatenate([grad_log_a_flat, grad_log_b])

    # We return the NEGATIVE log-likelihood & its gradient, because we minimize
    print(f"Current Loss: {-log_likelihood:.6f}")
    return -log_likelihood, -grad

def estimate_parameters(X, max_iter=100, save_dir="./parameter_estimation"):
    """
    Estimate a_{k,j}, b_k by maximizing the conditional log-likelihood.
    Saves results to estimated_parameters.pkl plus diagnostic plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    N, K = X.shape
    # Optionally use fewer samples to speed up (e.g. 8000 or fewer)
    N_train = min(N, 8000)
    X_train = X[:N_train]

    num_a_params = K*(K-1)
    num_b_params = K
    initial_params = np.random.randn(num_a_params + num_b_params) * 0.01

    objective_history = []

    def callback(params):
        obj_val, _ = conditional_log_likelihood(params, X_train)
        objective_history.append(obj_val)
        if len(objective_history) % 5 == 0:
            print(f"Iteration {len(objective_history)}: Loss = {obj_val:.6f}")

    print("Starting Optimization...")

    result = minimize(
        lambda p: conditional_log_likelihood(p, X_train),
        initial_params,
        method='L-BFGS-B',
        jac=True,
        callback=callback,
        options={'maxiter': max_iter, 'disp': True}
    )

    if result.success:
        print(f"Optimization successful! Final loss: {result.fun:.6f}")
    else:
        print(f"WARNING: Optimization did not converge. Message: {result.message}")

    optimized_params = result.x
    log_a_flat = optimized_params[:num_a_params]
    log_b = optimized_params[num_a_params:]

    # Reshape
    a = np.zeros((K, K))
    idx = 0
    for k in range(K):
        for j in range(K):
            if j != k:
                a[k, j] = np.exp(log_a_flat[idx])
                idx += 1
    b = np.exp(log_b)

    # Save
    params_data = {
        'a': a,
        'b': b,
        'optimized_params': optimized_params,
        'objective_history': objective_history
    }
    save_data_for_plotting(params_data, os.path.join(save_dir, 'estimated_parameters.pkl'))

    # Plot objective function
    plt.figure(figsize=(10, 6))
    plt.plot(objective_history, marker='o', linestyle='-')
    plt.title('Negative Conditional Log-Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'convergence.png'))
    plt.close()

    # Visualize the a matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(a, cmap='viridis')
    plt.colorbar(label='a[k,j] values')
    plt.title('Estimated a[k,j] Parameters')
    plt.xlabel('Component j')
    plt.ylabel('Component k')
    plt.savefig(os.path.join(save_dir, 'parameter_a_matrix.png'))
    plt.close()

    # Visualize b parameters
    plt.figure(figsize=(10, 5))
    plt.plot(b, marker='o')
    plt.title('Estimated b[k] Parameters')
    plt.xlabel('Component k')
    plt.ylabel('b[k]')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'parameter_b_values.png'))
    plt.close()

    return a, b

def main():
    # Adjust path if needed for your HPC
    mat_file_path = "./representational.mat"
    output_dir = "./representational_output_part2"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {mat_file_path} ...")
    mat_data = scipy.io.loadmat(mat_file_path)
    # Extract variables
    Y = mat_data['Y']  # shape (32000, 1024)
    R = mat_data['R']  # shape (1024, 256)
    W = mat_data['W']  # shape (1024, 256)

    # Compute X
    X = Y @ R  # shape (32000, 256)
    print("Data shape:", X.shape)

    # Estimate parameters
    print("\n*** Estimating conditional model parameters (Question 2) ***")
    a, b = estimate_parameters(
        X, 
        max_iter=100,  # you can reduce this if it's too slow
        save_dir=os.path.join(output_dir, "parameter_estimation")
    )
    print("\nDone estimating parameters!")

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.io
import os
import pickle

def save_data_for_plotting(data, filename):
    """Save data to file for later plotting locally"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def conditional_log_likelihood(params, X):
    """
    Compute the negative conditional log-likelihood and its gradient.
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

    # Convert to actual parameters
    a = np.exp(log_a)
    b = np.exp(log_b)

    # Initialize log-likelihood and gradients
    log_likelihood = 0.0
    grad_log_a = np.zeros_like(log_a)
    grad_log_b = np.zeros_like(log_b)

    # Compute log-likelihood and derivatives
    for n in range(N):
        if n % 1000 == 0:  # Print progress less frequently to reduce output file size
            print(f"Processing sample {n}/{N}...")

        for k in range(K):
            sigma_squared = np.sum(a[k, :] * X[n, :]**2) - a[k, k] * X[n, k]**2 + b[k]

            # Debugging check for NaN or Inf values
            if not np.isfinite(sigma_squared) or sigma_squared <= 0:
                print(f"WARNING: sigma_squared is invalid at (n={n}, k={k}). Value: {sigma_squared}")
                sigma_squared = 1e-6  # Prevent division by zero

            x_nk = X[n, k]
            log_likelihood += -0.5 * np.log(2*np.pi) - 0.5 * np.log(sigma_squared) - x_nk**2 / (2 * sigma_squared)

            for j in range(K):
                if j != k:
                    grad_log_a[k, j] += (-0.5 / sigma_squared + 0.5 * (x_nk**2 / sigma_squared**2)) * X[n, j]**2 * a[k, j]

            grad_log_b[k] += (-0.5 / sigma_squared + 0.5 * (x_nk**2 / sigma_squared**2)) * b[k]

    # Flatten gradients
    grad_log_a_flat = np.zeros(K*(K-1))
    idx = 0
    for k in range(K):
        for j in range(K):
            if j != k:
                grad_log_a_flat[idx] = grad_log_a[k, j]
                idx += 1

    grad = np.concatenate([grad_log_a_flat, grad_log_b])

    # Print loss value less frequently
    print(f"Current Loss: {-log_likelihood:.6f}")

    return -log_likelihood, -grad

def estimate_parameters(X, max_iter=100, save_dir="./output_data"):
    """
    Estimate parameters of the conditional model.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    N, K = X.shape
    N_train = min(N, 8000)  # Use 8000 samples as suggested in the question
    X_train = X[:N_train]

    num_a_params = K * (K - 1)
    num_b_params = K
    initial_params = np.random.randn(num_a_params + num_b_params) * 0.01

    # Track optimization progress
    objective_history = []

    def callback(params):
        obj_val = conditional_log_likelihood(params, X_train)[0]
        objective_history.append(obj_val)
        if len(objective_history) % 5 == 0:  # Print every 5 iterations
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

    # Extract optimized parameters
    optimized_params = result.x
    log_a_flat = optimized_params[:K*(K-1)]
    log_b = optimized_params[K*(K-1):]

    a = np.zeros((K, K))
    idx = 0
    for k in range(K):
        for j in range(K):
            if j != k:
                a[k, j] = np.exp(log_a_flat[idx])
                idx += 1

    b = np.exp(log_b)
    
    # Save parameters
    params_data = {
        'a': a,
        'b': b,
        'optimized_params': optimized_params,
        'objective_history': objective_history
    }
    save_data_for_plotting(params_data, os.path.join(save_dir, 'estimated_parameters.pkl'))

    # Plot and save objective function to verify convergence
    plt.figure(figsize=(10, 6))
    plt.plot(objective_history, marker='o', linestyle='-', markersize=3)
    plt.title('Negative Conditional Log-Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'convergence.png'))
    plt.close()  # Close to save memory

    # Visualize parameter matrix a
    plt.figure(figsize=(12, 10))
    plt.imshow(a, cmap='viridis')
    plt.colorbar(label='a[k,j] values')
    plt.title('Visualization of the a[k,j] parameters')
    plt.xlabel('Component j')
    plt.ylabel('Component k')
    plt.savefig(os.path.join(save_dir, 'parameter_a_matrix.png'))
    plt.close()  # Close to save memory
    
    # Visualize b parameters
    plt.figure(figsize=(10, 6))
    plt.plot(b, marker='o', linestyle='-')
    plt.title('b[k] parameters')
    plt.xlabel('Component k')
    plt.ylabel('b[k] value')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'parameter_b_values.png'))
    plt.close()  # Close to save memory

    return a, b

def analyze_components(X, W, save_dir="./output_data"):
    """
    Analyze the components for sparsity and independence.
    This addresses question 1 in the assignment.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    N, K = X.shape
    
    # Select some components to analyze, including high frequency ones
    # Select components at regular intervals plus some extra at the end (likely higher frequency)
    selected_components = list(range(0, K, 25)) + [K-5, K-3, K-1]
    selected_components = sorted(list(set(selected_components)))  # Remove duplicates
    
    # 1(i) Plot marginal distributions
    hist_data = {}
    
    for k in selected_components:
        # Compute histogram
        hist, bin_edges = np.histogram(X[:, k], bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Save histogram data
        hist_data[k] = {'hist': hist, 'bin_centers': bin_centers, 'values': X[:, k]}
        
        # Plot on linear scale
        plt.figure(figsize=(10, 6))
        plt.hist(X[:, k], bins=100, density=True, alpha=0.7)
        plt.title(f'Marginal Distribution of Component {k}')
        plt.xlabel('Component Value')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'component_{k}_distribution_linear.png'))
        plt.close()
        
        # Plot on log scale to check for sparsity
        plt.figure(figsize=(10, 6))
        non_zero_hist = hist.copy()
        non_zero_hist[non_zero_hist <= 0] = 1e-10  # Avoid log(0)
        plt.plot(bin_centers, np.log(non_zero_hist))
        
        # Add Gaussian with same mean and variance for comparison
        mean = np.mean(X[:, k])
        std = np.std(X[:, k])
        x_range = np.linspace(min(bin_centers), max(bin_centers), 1000)
        log_gaussian = -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * ((x_range - mean) / std) ** 2
        plt.plot(x_range, log_gaussian, 'r--', label='Gaussian with same mean & variance')
        
        plt.title(f'Log Marginal Distribution of Component {k}')
        plt.xlabel('Component Value')
        plt.ylabel('Log Probability Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'component_{k}_distribution_log.png'))
        plt.close()
    
    # Save all histogram data
    save_data_for_plotting(hist_data, os.path.join(save_dir, 'histogram_data.pkl'))
    
    # 1(ii) Analyze pairwise dependencies
    pair_data = {}
    
    # Find pairs with similar location and orientation by analyzing W
    # For this, we might pair components that have high correlation in their weights
    W_corr = np.corrcoef(W.T)
    np.fill_diagonal(W_corr, 0)  # Zero out diagonal
    
    # Find top correlated pairs
    flat_corr = W_corr.flatten()
    top_indices = np.argsort(np.abs(flat_corr))[-20:]  # 20 highest correlated pairs
    correlated_pairs = []
    
    for idx in top_indices:
        i, j = idx // K, idx % K
        if i < j:  # Avoid duplicates
            correlated_pairs.append((i, j))
    
    # Add some random pairs and specified pairs for comparison
    pairs_to_analyze = correlated_pairs[:5]  # Top 5 correlated pairs
    pairs_to_analyze.extend([(k1, k2) for k1, k2 in zip(selected_components[:-1], selected_components[1:])])
    
    for k1, k2 in pairs_to_analyze:
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(X[:, k1], X[:, k2], bins=50, density=True)
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        
        # Save 2D histogram data
        pair_data[(k1, k2)] = {
            'H': H, 
            'xcenters': xcenters, 
            'ycenters': ycenters,
            'values_k1': X[:, k1],
            'values_k2': X[:, k2],
            'corr': np.corrcoef(X[:, k1], X[:, k2])[0, 1]
        }
        
        # Plot 2D histogram
        plt.figure(figsize=(10, 8))
        plt.imshow(H.T, aspect='auto', origin='lower', 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar(label='Joint Probability Density')
        plt.title(f'Joint Distribution p(x_{k1}, x_{k2}), Corr: {pair_data[(k1, k2)]["corr"]:.3f}')
        plt.xlabel(f'Component {k1}')
        plt.ylabel(f'Component {k2}')
        plt.savefig(os.path.join(save_dir, f'joint_dist_{k1}_{k2}.png'))
        plt.close()
        
        # Create conditional distributions p(x_k2|x_k1)
        H_norm = H.copy()
        for i in range(H.shape[0]):
            row_sum = H[i, :].sum()
            if row_sum > 0:
                H_norm[i, :] /= row_sum
            else:
                H_norm[i, :] = 0
        
        # Plot conditional distribution
        plt.figure(figsize=(10, 8))
        plt.imshow(H_norm.T, aspect='auto', origin='lower', 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar(label='Conditional Probability p(x_k2|x_k1)')
        plt.title(f'Conditional Distribution p(x_{k2}|x_{k1})')
        plt.xlabel(f'Component {k1}')
        plt.ylabel(f'Component {k2}')
        plt.savefig(os.path.join(save_dir, f'conditional_dist_{k1}_{k2}.png'))
        plt.close()
    
    # Save all pair data
    save_data_for_plotting(pair_data, os.path.join(save_dir, 'pair_data.pkl'))
    
    # Also save the correlation matrix of W for reference
    plt.figure(figsize=(12, 10))
    plt.imshow(np.abs(W_corr), cmap='viridis')
    plt.colorbar(label='Absolute Correlation')
    plt.title('Absolute Correlation Between Generative Weights')
    plt.xlabel('Component j')
    plt.ylabel('Component k')
    plt.savefig(os.path.join(save_dir, 'W_correlation_matrix.png'))
    plt.close()
    
    save_data_for_plotting({'W_corr': W_corr}, os.path.join(save_dir, 'W_correlation.pkl'))
    
    return hist_data, pair_data

def main():
    output_dir = "./representational_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the .mat file - UPDATE THIS PATH for your HPC environment
    mat_file_path = "./representational.mat"
    
    # Load the .mat file
    print(f"Loading data from {mat_file_path}...")
    mat_data = scipy.io.loadmat(mat_file_path)

    # Extract variables
    Y = mat_data['Y']  # 32000 × 1024 image patches
    R = mat_data['R']  # 1024 × 256 feed-forward weights
    W = mat_data['W']  # 1024 × 256 generative weights
    X = Y @ R
    print(f"Data shape: {X.shape}")
    
    # Question 1: Analyze components
    print("\n*** Analyzing components for sparsity and independence (Question 1) ***")
    hist_data, pair_data = analyze_components(X, W, save_dir=os.path.join(output_dir, "component_analysis"))
    
    # Question 2: Estimate conditional model parameters
    print("\n*** Estimating conditional model parameters (Question 2) ***")
    a, b = estimate_parameters(X, max_iter=100, save_dir=os.path.join(output_dir, "parameter_estimation"))
    
    print("\nAll analyses completed. Results saved to:", output_dir)
    
    # Create a summary file
    summary = {
        'data_shape': X.shape,
        'n_samples': X.shape[0],
        'n_components': X.shape[1],
        'output_directory': os.path.abspath(output_dir)
    }
    
    with open(os.path.join(output_dir, "analysis_summary.txt"), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()
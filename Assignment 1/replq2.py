import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.io


def conditional_log_likelihood(params, X):

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
        for k in range(K):
            sigma_squared = np.sum(a[k, :] * X[n, :]**2) - a[k, k] * X[n, k]**2 + b[k]
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
    
    return -log_likelihood, -grad

def estimate_parameters(X, max_iter=100):

    """
    Estimate parameters of the conditional model.
    """

    N, K = X.shape
    N_train = min(N, 8000)
    X_train = X[:N_train]
    
    num_a_params = K * (K - 1)
    num_b_params = K
    initial_params = np.random.randn(num_a_params + num_b_params) * 0.01
    
    # Optimize parameters
    objective_history = []

    def callback(params):
        obj_val = conditional_log_likelihood(params, X_train)[0]
        objective_history.append(obj_val)

    result = minimize(
        lambda p: conditional_log_likelihood(p, X_train),
        initial_params,
        method='L-BFGS-B',
        jac=True,
        callback=callback,
        options={'maxiter': max_iter, 'disp': True}
    )

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
    
    # Plot objective function to verify convergence
    plt.figure(figsize=(8, 5))
    plt.plot(objective_history, marker='o', linestyle='-')
    plt.title('Negative Conditional Log-Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.savefig('convergence.png')
    plt.show()
    
    return a, b

def analyze_distribution(X, a, b):

    """
    Analyze the distribution and dependencies in the data.
    """
    N, K = X.shape
    k = np.random.randint(K)
    
    sigma_squared = np.zeros(N)
    for n in range(N):
        sigma_squared[n] = np.sum(a[k, :] * X[n, :]**2) - a[k, k] * X[n, k]**2 + b[k]

    plt.figure(figsize=(10, 5))

    # Plot histogram of actual data
    plt.subplot(2, 1, 1)
    plt.hist(X[:, k], bins=50, density=True, alpha=0.6, color='b', label='Actual')
    
    # Create a range of x values and plot the average conditional Gaussian
    x = np.linspace(np.min(X[:, k]), np.max(X[:, k]), 1000)
    avg_sigma = np.mean(sigma_squared)
    y = np.exp(-0.5 * x**2 / avg_sigma) / np.sqrt(2 * np.pi * avg_sigma)
    plt.plot(x, y, 'r', linewidth=2, label='Average Conditional Gaussian')
    
    plt.title(f'Distribution of x_{k}')
    plt.legend()

    # Plot on log scale to check sparsity
    plt.subplot(2, 1, 2)
    plt.hist(X[:, k], bins=50, density=True, alpha=0.6, color='b', label='Actual')
    plt.plot(x, y, 'r', linewidth=2, label='Average Conditional Gaussian')
    plt.yscale('log')
    plt.title(f'Log-scale Distribution of x_{k}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('distribution.png')
    plt.show()


def main():

    # Load the .mat file - adjust this path to where your file actually is
    mat_data = scipy.io.loadmat("representational/representational.mat")

    # Extract variables
    Y = mat_data['Y']  # 32000 × 1024 image patches
    R = mat_data['R']  # 1024 × 256 feed-forward weights
    X = Y @ R
    print(f"Data shape: {X.shape}")

    a, b = estimate_parameters(X[:8000,:], max_iter=20)
    analyze_distribution(X, a, b)

if __name__ == "__main__":
    main()

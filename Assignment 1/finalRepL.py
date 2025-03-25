#!/usr/bin/env python3
"""
Sparse Coding Model - Questions 3 & 4
-------------------------------------
This script illustrates how to:
 - Compute normalized variables c_{n,k} = x_{n,k} / sigma_k(x_{n,neq k})
 - Plot marginal & conditional distributions for x_{n,k} and c_{n,k}
 - Plot relevant generative weights and investigate top a_{k,j} connections.

Make sure to have the following variables loaded or computed beforehand:
 X  : shape (N, K)       -> latent (feed-forward) coefficients
 a  : shape (K, K)       -> learned parameters for variance dependence
 b  : shape (K,)         -> learned offsets
 W  : shape (D, K)       -> dictionary / generative weights (e.g., D=1024, K=256)
"""

#============================ Imports ============================#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

#============================ Utility Functions ============================#
def compute_sigma2(X, a, b):
    """
    Compute sigma^2_{n,k} = sum_{j != k} a[k,j] * X[n,j]^2 + b[k]
    for each sample n and each component k.
    
    Parameters
    ----------
    X : ndarray, shape (N, K)
        Latent variables x_{n,k}
    a : ndarray, shape (K, K)
        Quadratic dependence parameters a_{k,j}
    b : ndarray, shape (K,)
        Offset parameters b_k
    
    Returns
    -------
    sigma2 : ndarray, shape (N, K)
        The variance sigma^2_{n,k} for each n,k
    """
    N, K = X.shape
    sigma2 = np.zeros((N, K))
    
    # For each k, we sum over j != k
    # but a straightforward way is to sum over all j and subtract the j=k term if needed
    for k_ in range(K):
        # sum_{j} a[k_, j] * X^2[:, j]
        # then we add b[k_]
        # a[k_, k_] * X[:,k_]^2 is included, so we'll subtract that out (since j != k).
        sum_over_j = X**2 @ a[k_, :]  # shape (N,)
        sigma2[:, k_] = sum_over_j + b[k_]
        # remove the j=k_ term if you want strictly j != k:
        # sigma2[:, k_] -= a[k_, k_] * (X[:, k_]**2) 
        #
        # BUT if the learned model actually includes the k=j term (some implementations might),
        # then keep it as is. Check your exact formulation. 
        #
        # If you strictly want j != k, uncomment below line:
        sigma2[:, k_] -= a[k_, k_] * (X[:, k_]**2)
    
    return sigma2

def plot_hist_comparison(x_data, c_data, component_idx=0, num_bins=100):
    """
    Plot histograms for x_{n,k} and c_{n,k} side by side (on log y-scale),
    along with their kurtosis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot for X_{:,k}
    axes[0].hist(x_data, bins=num_bins, density=True, alpha=0.7, edgecolor='black')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('x_k')
    axes[0].set_ylabel('log probability')
    axes[0].set_title(f'X Dist (k={component_idx}), Excess Kurt={kurtosis(x_data):.2f}')
    
    # Plot for C_{:,k}
    axes[1].hist(c_data, bins=num_bins, density=True, alpha=0.7, edgecolor='black')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('c_k = x_k / sigma_k')
    axes[1].set_ylabel('log probability')
    axes[1].set_title(f'C Dist (k={component_idx}), Excess Kurt={kurtosis(c_data):.2f}')
    
    plt.tight_layout()
    plt.show()

def plot_conditional_2D(x1, x2, title='', bins=60):
    """
    Plot p(x2 | x1) by forming a 2D histogram (x1 on x-axis, x2 on y-axis)
    and normalizing each vertical slice for x1.
    """
    # 2D histogram
    H, xedges, yedges = np.histogram2d(x1, x2, bins=bins, density=False)
    
    # We want p(x2 | x1), so for each bin i in x1, we normalize over x2.
    # H[i, :] are the counts for a particular bin in x1 across all x2-bins.
    for i in range(H.shape[0]):
        row_sum = H[i, :].sum()
        if row_sum > 0:
            H[i, :] /= row_sum
    
    # Plot as an image
    # x1 -> horizontal axis, x2 -> vertical axis
    # note: imshow uses H[row, col], row=0 at top, so we either flip or transpose.
    # We'll use origin='lower' and the edges in ascending order to keep it intuitive.
    plt.figure(figsize=(6, 5))
    plt.imshow(
        H.T, 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower', 
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label='p(x2 | x1)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.show()

def plotIm(w, patch_size=32, title=''):
    """
    Display a 1D weight vector w (length patch_size^2) as a 2D grayscale image.
    Adjust patch_size or reshape logic as needed for your data.
    """
    plt.figure()
    plt.imshow(w.reshape((patch_size, patch_size)), cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

#============================ Main Script ============================#

def main():
    #------------------------------------------------------------------
    # Assume you have loaded or computed the following:
    #    X : (N, K) array of latent variables
    #    a : (K, K) array of parameters
    #    b : (K,)   array of parameters
    #    W : (D, K) dictionary, each column is one generative component
    #
    # Replace the placeholders below with your actual data loading code.
    #------------------------------------------------------------------
    
    # Example placeholders (DO NOT RUN as-is; fill in with real data):
    # X = np.load('X.npy')  # shape (N, K)
    # a = np.load('a.npy')  # shape (K, K)
    # b = np.load('b.npy')  # shape (K,)
    # W = np.load('W.npy')  # shape (D, K)
    
    # For demonstration, let's create some random data of suitable shapes:
    # (Comment this out and replace with real loaded data.)
    np.random.seed(0)
    N, K, D = 5000, 10, 1024  # smaller example
    X = 0.5 * np.random.randn(N, K)
    a = 0.01 * np.abs(np.random.randn(K, K))  # small positive
    b = 0.01 * np.ones(K)
    W = np.random.randn(D, K)
    
    #============================ Question 3 ============================#
    
    #----- 3.1: Compute c_{n,k} = x_{n,k} / sigma_k(x_{n,neq k};theta) -----#
    sigma2 = compute_sigma2(X, a, b)       # shape (N, K)
    sigma = np.sqrt(sigma2)                # shape (N, K)
    C = X / sigma                          # normalized latents c_{n,k}
    
    #----- 3.2: Plot the marginal distribution p(x_k) and p(c_k) -----#
    # Choose a few components to illustrate:
    chosen_components = [0, 1, 2]  # or any interesting subset
    
    for k_ in chosen_components:
        x_k = X[:, k_]
        c_k = C[:, k_]
        plot_hist_comparison(x_k, c_k, component_idx=k_, num_bins=80)
    
    #----- 3.3: Compare conditional distributions p(x_{k2}|x_{k1}) vs. p(c_{k2}|c_{k1}) -----#
    # We'll pick a pair of components, e.g. (k1, k2) = (0, 1)
    k1, k2 = 0, 1
    
    x1 = X[:, k1]
    x2 = X[:, k2]
    c1 = C[:, k1]
    c2 = C[:, k2]
    
    # Plot 2D conditional histogram for X
    plot_conditional_2D(
        x1, 
        x2, 
        title=f'Conditional distribution p(x_{k2} | x_{k1})',
        bins=50
    )
    
    # Plot 2D conditional histogram for C
    plot_conditional_2D(
        c1, 
        c2, 
        title=f'Conditional distribution p(c_{k2} | c_{k1})',
        bins=50
    )
    
    # You can visually compare whether p(x_{k2} | x_{k1}) is more or less "independent-like"
    # compared to p(c_{k2} | c_{k1}). Also check changes in structure.
    
    
    #============================ Question 4 ============================#
    
    #----- 4.1: Pick a component k and plot its generative weight -----#
    comp_k = 0  # choose whichever component you like
    plotIm(
        W[:, comp_k], 
        patch_size=32, 
        title=f'Generative Weight W[:,{comp_k}]'
    )
    
    #----- 4.2: Plot the weights of the top 10 components for which a_{k,j} is largest -----#
    # We find the largest a_{k, j} across j (excluding j=k if you wish).
    a_row = a[comp_k, :]  # shape (K,)
    # If you want to exclude comp_k itself:
    # a_row[comp_k] = -np.inf
    top_10_j = np.argsort(a_row)[-10:]  # indices of top 10
    print(f"Top 10 j indices for a[k={comp_k}, j]: {top_10_j}")
    
    for j_idx in top_10_j:
        plotIm(
            W[:, j_idx], 
            patch_size=32, 
            title=f'Generative Weight W[:,{j_idx}] (large a[{comp_k},{j_idx}])'
        )
    
    # Similarly, you could repeat for other components comp_k = 1, 2, etc.
    # Observe whether these top connections correspond to similar edge/orientation, etc.

if __name__ == "__main__":
    main()

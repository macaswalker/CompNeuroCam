#%%
import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from scripts import plotIm, minimize, checkgrad
#%%

#============================ Data Loading =============================#
# Print current working directory to see where Python is looking
print(f"Current working directory: {os.getcwd()}")

# Load the .mat file - adjust this path to where your file actually is
mat_data = scipy.io.loadmat("representational/representational.mat")
#%%

#============================ EDA =============================#

# Extract variables
Y = mat_data['Y']  # 32000 × 1024 image patches
R = mat_data['R']  # 1024 × 256 feed-forward weights
W = mat_data['W']  # Another 1024 × 256 matrix
X = Y @ R # (32000 x 1024) x (1024 x 256) = (32000, 256)

print(f"Y shape: {Y.shape}")
print(f"R shape: {R.shape}")
print(f"W shape: {W.shape}")

# plotIm(W) # plot all of the generative weight filters

#%%

#============================ Question 1 =============================#

#====================== 1.i =======================#
X = Y @ R # (32000 x 1024) x (1024 x 256) = (32000, 256)
print(f"X shape: {X.shape}")

# Select a few components to analyze (including high-frequency ones)
selected_components = [125, 129, 133, 137, 141]  # Example indices of columns in X

# Plot histograms for selected components
fig, axes = plt.subplots(1, len(selected_components), figsize=(15, 4))

for i, k in enumerate(selected_components):
    axes[i].hist(X[:, k], bins=50, density=True, alpha=0.7, label=f"Component {k}")
    
    # LaTeX-style titles and axis labels
    axes[i].set_title(f"Histogram of $x_{{{k}}}$")  # LaTeX for x_k
    axes[i].set_xlabel(f"$x_{{{k}}}$ values (bins = 50)")  # r"" ensures raw string for LaTeX
    axes[i].set_ylabel(r"Density")  # Keep this standard
    axes[i].set_yscale('log')  # Log scale for better visualization

plt.tight_layout()
plt.show()
#%%

# #====================== 1.ii =======================#
#plot the 30th and 31st 32x32 matric from W

#Create a 2x1 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 12))

# Plot the 30th and 31st components of W
for i, k in enumerate([125, 141]):    
    patch = W[:, k].reshape(32, 32)
    axes[i].imshow(patch, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Component {k}', pad=10)  # Add title with some padding

plt.tight_layout()
plt.show()  

#%%

# let us cimpare compents 233 and 249, using a 2d histogram

#Create a 2D histogram of X[:, 233] vs X[:, 249]
plt.figure(figsize=(8, 6))
plt.hist2d(X[:, 125], X[:, 141], bins=50, cmap='viridis', norm=colors.LogNorm(vmax=1e4, vmin=1e-4))  # Add LogNorm
plt.colorbar(label='Frequency (log scale)')
plt.xlabel('Component 125')   
plt.ylabel('Component 141')
plt.title('2D Histogram of Component 125 vs Component 141 (log scale)')
plt.show()  
#%%

#compute and plot the conditional distribution p(x_k2|x_k1)
H, x_edges, y_edges = np.histogram2d(X[:, 125], X[:, 141], bins=50)

# Normalize each column (x_k1 slice) to get conditional probability
# Add small constant to avoid division by zero
H_conditional = H / (H.sum(axis=0, keepdims=True) + 1e-10)

# Plot the conditional distribution
plt.figure(figsize=(8, 6))
plt.pcolormesh(x_edges[:-1], y_edges[:-1], H_conditional.T, cmap='viridis', norm=colors.LogNorm(vmax=1e4, vmin=1e-4))  # Add LogNorm
plt.colorbar(label='$p(x_{k2}|x_{k1})$ (log scale)')
plt.xlabel('$x_{k1}$ (Component 125)')
plt.ylabel('$x_{k2}$ (Component 141)')
plt.title('Conditional Distribution $p(x_{k2}|x_{k1})$ (log scale)')
plt.show()

#============================ Question 2 =============================#
#%% [markdown]
# #============================ Question 3 ============================#
# 
# We now load the learned parameters (a, b) from the .pkl file instead of
# using random placeholders. Then, we'll compute the normalized coefficients
# c = x / sigma(x) and plot the marginal & conditional distributions.

#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import kurtosis

# Path to your trained parameter checkpoint
pkl_path = "/Users/mac/Documents/Cambridge/Lent/Comp Neuro/Assignments/Assignment 1/HPC_Coursework/representational_output_fast/parameter_estimation_checkpoint/estimated_parameters.pkl"

# Load the dictionary containing 'a' and 'b'
with open(pkl_path, "rb") as f:
    param_dict = pickle.load(f)

# Extract the arrays for 'a' and 'b'. Adjust the keys if your .pkl uses different names.
a_data = param_dict["a"]  # shape should be (K, K)
b_data = param_dict["b"]  # shape should be (K,)

print("Loaded parameters from pickle:")
print("a_data shape:", a_data.shape)
print("b_data shape:", b_data.shape)

# Confirm that X exists (from your code above, X = Y @ R, shape: (N, K))
print("X shape:", X.shape)

#--------------------------------------------------------------------------------
def compute_sigma2(X, a, b):
    """
    Compute sigma^2_{n,k} = sum_{j != k} a[k,j]*X[n,j]^2 + b[k]
    for each sample n and each component k.
    """
    N, K = X.shape
    sigma2 = np.zeros((N, K))
    for k_ in range(K):
        # sum_{j} a[k_, j] * X^2[:, j]
        sum_over_j = (X**2) @ a[k_, :]
        sigma2[:, k_] = sum_over_j + b[k_]
        # Remove diagonal term if your model excludes j = k
        sigma2[:, k_] -= a[k_, k_] * (X[:, k_]**2)
    return sigma2

# Compute sigma^2(x) and then normalized coefficients c
sigma2_vals = compute_sigma2(X, a_data, b_data)
sigma_vals = np.sqrt(sigma2_vals)  # shape (N, K)
C = X / sigma_vals

print("Computed normalized coefficients C = X / sigma(x).")
print("C shape:", C.shape)

#------------------------- 3.1: Marginal Distributions -------------------------#
# Compare p(x_k) and p(c_k) for some chosen components
chosen_components = [125, 141]  # pick whichever indices are interesting
fig, axes = plt.subplots(2, len(chosen_components), figsize=(4*len(chosen_components), 8))

for idx, k_ in enumerate(chosen_components):
    # Plot X_k
    axes[0, idx].hist(X[:, k_], bins=50, density=True, alpha=0.7, edgecolor='k')
    axes[0, idx].set_yscale('log')
    axes[0, idx].set_xlabel(f"$x_{{{k_}}}$")
    axes[0, idx].set_ylabel("Density (log-scale)")
    kurt_x = kurtosis(X[:, k_])  # Excess kurtosis
    axes[0, idx].set_title(f"X_{k_}, Kurt={kurt_x:.2f}")

    # Plot C_k
    axes[1, idx].hist(C[:, k_], bins=50, density=True, alpha=0.7, edgecolor='k')
    axes[1, idx].set_yscale('log')
    axes[1, idx].set_xlabel(f"$c_{{{k_}}}$ = x_{{{k_}}}/\\sigma_{{{k_}}}$")
    axes[1, idx].set_ylabel("Density (log-scale)")
    kurt_c = kurtosis(C[:, k_])
    axes[1, idx].set_title(f"C_{k_}, Kurt={kurt_c:.2f}")

plt.tight_layout()
plt.show()

#--------------------- 3.2: Conditional Distributions ---------------------#
def plot_conditional_2D(x1, x2, title='', bins=60, lognorm=True):
    """
    Plot p(x2 | x1) by forming a 2D histogram and normalizing each vertical slice.
    """
    H, xedges, yedges = np.histogram2d(x1, x2, bins=bins, density=False)
    for i in range(H.shape[0]):
        col_sum = np.sum(H[i, :])
        if col_sum > 0:
            H[i, :] /= col_sum
    plt.figure(figsize=(6, 5))
    if lognorm:
        plt.pcolormesh(xedges, yedges, H.T, cmap='viridis',
                       norm=colors.LogNorm(vmin=1e-5, vmax=H.max()))
    else:
        plt.pcolormesh(xedges, yedges, H.T, cmap='viridis')
    plt.colorbar(label="$p(x_2 \\mid x_1)$")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()

# Example pair: (k1=125, k2=141) or whichever pair you like
k1, k2 = 125, 141

# p(x_{k2} | x_{k1})
plot_conditional_2D(X[:, k1], X[:, k2], title=f"p(x_{k2} | x_{k1}) (Components {k1} vs. {k2})")

# p(c_{k2} | c_{k1})
plot_conditional_2D(C[:, k1], C[:, k2], title=f"p(c_{k2} | c_{k1}) (Components {k1} vs. {k2})")

#%% [markdown]
# #============================ Question 4 ============================#
# 
# 1. Pick a component k, plot its generative weight W[:,k].
# 2. Find the top-10 components j for which a[k, j] is largest, and plot W[:,j].
#    This shows the strongest dependencies in the learned model.

#%%

k = 125

# First, plot the main filter W[:,k] in its own figure
main_filter_img = W[:, k].reshape(32, 32)
plt.figure()
plt.imshow(main_filter_img, cmap='gray')
plt.title(f"Generative Weight W[:, {k}]")
plt.axis('off')
plt.show()

# Find top 10 largest a[k, j]
row_k = a_data[k, :]
top_10_indices = np.argsort(row_k)[-10:]
print(f"Top 10 j indices for a[k={k}, j]: {top_10_indices}")
print("Corresponding a-values:", row_k[top_10_indices])

# Plot those top-10 filters in a single 2×5 figure
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for idx, j_ in enumerate(top_10_indices):
    # Map idx=0..9 to (row, col) in a 2×5 grid
    row = idx // 5  # 0 for idx=0..4, 1 for idx=5..9
    col = idx % 5   # cycles through 0..4
    
    # Reshape and plot
    img = W[:, j_].reshape(32, 32)
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f"W[:,{j_}]  a[{k},{j_}]={row_k[j_]:.4f}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()




# %%
k = 162

# First, plot the main filter W[:,k] in its own figure
main_filter_img = W[:, k].reshape(32, 32)
plt.figure()
plt.imshow(main_filter_img, cmap='gray')
plt.title(f"Generative Weight W[:, {k}]")
plt.axis('off')
plt.show()

# Find top 10 largest a[k, j]
row_k = a_data[k, :]
top_10_indices = np.argsort(row_k)[-10:]
print(f"Top 10 j indices for a[k={k}, j]: {top_10_indices}")
print("Corresponding a-values:", row_k[top_10_indices])

# Plot those top-10 filters in a single 2×5 figure
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for idx, j_ in enumerate(top_10_indices):
    # Map idx=0..9 to (row, col) in a 2×5 grid
    row = idx // 5  # 0 for idx=0..4, 1 for idx=5..9
    col = idx % 5   # cycles through 0..4
    
    # Reshape and plot
    img = W[:, j_].reshape(32, 32)
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f"W[:,{j_}]  a[{k},{j_}]={row_k[j_]:.4f}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
# %%

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Use the full absolute path
dir_path = "/Users/mac/Documents/Cambridge/Lent/Comp Neuro/Assignments/Assignment 1/HPC_Coursework/representational_output_fast/parameter_estimation_checkpoint"

# List files in the directory to verify
print("Files in directory:")
print(os.listdir(dir_path))

# Load the pickle files
try:
    with open(os.path.join(dir_path, 'partial_params_last.pkl'), 'rb') as f:
        partial_params = pickle.load(f)

    print("\nPartial Parameters:")
    for key, value in partial_params.items():
        print(f"{key}: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            print(f"  First few values: {value[:5]}")
        else:
            print(f"  Value: {value}")

except Exception as e:
    print(f"Error loading partial_params: {e}")

try:
    with open(os.path.join(dir_path, 'estimated_parameters.pkl'), 'rb') as f:
        estimated_params = pickle.load(f)

    print("\nEstimated Parameters:")
    for key, value in estimated_params.items():
        print(f"{key}: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            print(f"  First few values: {value[:5]}")
        else:
            print(f"  Value: {value}")

except Exception as e:
    print(f"Error loading estimated_params: {e}")

# Visualization of 'a' matrix
plt.figure(figsize=(20, 8))

# Partial Parameters 'a' matrix
plt.subplot(1, 2, 1)
im1 = plt.imshow(partial_params['a'], cmap='viridis', aspect='auto')
plt.colorbar(im1)
plt.title('Partial Parameters - a Matrix')
plt.xlabel('Component j')
plt.ylabel('Component k')

# Estimated Parameters 'a' matrix
plt.subplot(1, 2, 2)
im2 = plt.imshow(estimated_params['a'], cmap='viridis', aspect='auto')
plt.colorbar(im2)
plt.title('Estimated Parameters - a Matrix')
plt.xlabel('Component j')
plt.ylabel('Component k')

plt.tight_layout()
plt.savefig('parameter_a_comparison.png')
plt.close()

# Separate plot for 'b' parameters
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(partial_params['b'])
plt.title('Partial Parameters - b Values')
plt.xlabel('Component')
plt.ylabel('b Value')

plt.subplot(1, 2, 2)
plt.plot(estimated_params['b'])
plt.title('Estimated Parameters - b Values')
plt.xlabel('Component')
plt.ylabel('b Value')

plt.tight_layout()
plt.savefig('parameter_b_comparison.png')
plt.close()

# Print some basic statistics
print("Partial Parameters 'a' matrix:")
print(f"Mean: {np.mean(partial_params['a']):.4f}")
print(f"Standard Deviation: {np.std(partial_params['a']):.4f}")

print("\nEstimated Parameters 'a' matrix:")
print(f"Mean: {np.mean(estimated_params['a']):.4f}")
print(f"Standard Deviation: {np.std(estimated_params['a']):.4f}")
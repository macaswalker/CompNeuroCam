import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def plotIm(W, num_patches=None):
    """
    Plots square images reshaped from the generative weight matrix W.

    Parameters:
    W (numpy.ndarray): The generative weights matrix of shape (D, K).
    num_patches (int, optional): Number of patches to display (defaults to all K).
    """
    
    D, K = W.shape  # D: Number of pixels, K: Number of components
    NumPix = int(np.sqrt(D))  # Assuming square patches
    
    if NumPix ** 2 != D:
        raise ValueError("W's rows must be perfect squares (e.g., 32x32 patches).")

    # Determine grid size for displaying patches
    NumPlots = int(np.ceil(np.sqrt(K))) if num_patches is None else int(np.ceil(np.sqrt(num_patches)))
    
    # Select a subset if num_patches is specified
    W_subset = W[:, :num_patches] if num_patches else W

    # Set up figure and axes
    fig, axes = plt.subplots(NumPlots, NumPlots, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        if i < W_subset.shape[1]:
            patch = W_subset[:, i].reshape(NumPix, NumPix)  # Reshape to square
            ax.imshow(patch, cmap='gray', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
        ax.axis("off")

    plt.suptitle("Generative Weights Visualization", fontsize=14)
    plt.show()



def minimize(X, func, num_iterations, *args):
    """
    Minimize a differentiable multivariate function using conjugate gradients.

    Parameters:
    - X: numpy array, initial guess.
    - func: function to minimize, must return value and gradient.
    - num_iterations: int, max iterations (if negative, max function evaluations).
    - *args: additional parameters passed to func.

    Returns:
    - X: optimized variables.
    - fX: list of function values indicating progress.
    - i: number of iterations used.
    """
    
    # Constants for Wolfe-Powell line search conditions
    INT = 0.1   # Don't reevaluate too close to the boundary
    EXT = 3.0   # Max extrapolation factor
    MAX = 20    # Max line search evaluations per iteration
    RATIO = 10  # Max slope ratio
    SIG = 0.1   # Wolfe-Powell condition for sufficient decrease
    RHO = SIG / 2  # Wolfe-Powell condition for curvature

    # Initial function evaluation
    i = 0  # Iteration count
    ls_failed = 0  # Track line search failures
    f0, df0 = func(X, *args)  # Compute function value and gradient

    fX = [f0]  # Track function values
    s = -df0  # Initial search direction (steepest descent)
    d0 = -np.dot(s.T, s)  # Initial slope
    x3 = 1.0 / (1 - d0)  # Initial step size
    
    while i < abs(num_iterations):  # Main loop
        i += 1  # Increment iteration count
        
        X0, F0, dF0 = X.copy(), f0, df0.copy()  # Store current state
        M = MAX if num_iterations > 0 else min(MAX, -num_iterations - i)
        
        while True:  # Extrapolation loop
            x2, f2, d2 = 0, f0, d0
            success = False
            while not success and M > 0:
                try:
                    M -= 1
                    i += int(num_iterations < 0)  # Count epochs if needed
                    X_new = X + x3 * s
                    f3, df3 = func(X_new, *args)
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3) + np.isinf(df3)):
                        raise ValueError("Numerical issue in function evaluation")
                    success = True
                except:
                    x3 = (x2 + x3) / 2  # Bisect step size if failure
            
            if f3 < F0:
                X0, F0, dF0 = X_new.copy(), f3, df3.copy()
            
            d3 = np.dot(df3.T, s)  # New slope
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
                break  # Exit extrapolation
            
            x1, f1, d1 = x2, f2, d2
            x2, f2, d2 = x3, f3, d3
            
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            x3 = x1 - d1 * (x2 - x1) ** 2 / (B + np.sqrt(B * B - A * d1 * (x2 - x1)))
            
            if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:
                x3 = x2 * EXT  # Max extrapolation
        
        while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:  # Interpolation
            if d3 > 0 or f3 > f0 + x3 * RHO * d0:
                x4, f4, d4 = x3, f3, d3
            else:
                x2, f2, d2 = x3, f3, d3

            if f4 > f0:
                x3 = x2 - 0.5 * d2 * (x4 - x2) ** 2 / (f4 - f2 - d2 * (x4 - x2))
            else:
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A
            
            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2
            
            x3 = max(min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 - x2))
            X_new = X + x3 * s
            f3, df3 = func(X_new, *args)

            if f3 < F0:
                X0, F0, dF0 = X_new.copy(), f3, df3.copy()
            
            M -= 1
            i += int(num_iterations < 0)
            d3 = np.dot(df3.T, s)
        
        if abs(d3) < -SIG * d0 and f3 < f0 + x3 * RHO * d0:  # If line search succeeded
            X = X + x3 * s
            f0, df0 = f3, df3
            fX.append(f0)

            s = (np.dot(df3.T, df3) - np.dot(df0.T, df3)) / np.dot(df0.T, df0) * s - df3
            d0, d3 = np.dot(df0.T, s), d0
            
            if d0 > 0:  # If slope is positive, revert to steepest descent
                s = -df0
                d0 = -np.dot(s.T, s)
            
            x3 *= min(RATIO, d3 / (d0 + 1e-8))  # Adjust step size
            ls_failed = 0
        else:
            X, f0, df0 = X0.copy(), F0, dF0.copy()
            if ls_failed or i > abs(num_iterations):
                break  # If line search failed twice, exit
            
            s = -df0
            d0 = -np.dot(s.T, s)
            x3 = 1 / (1 - d0)
            ls_failed = 1  # Mark as line search failure

    return X, fX, i



def checkgrad(f, X, e=1e-4, *args):
    """
    Check the gradient of a function using finite differences.
    
    Parameters:
    - f: function that returns (function value, gradient).
    - X: numpy array, the point at which to check the gradient.
    - e: small perturbation for finite differences.
    - *args: additional arguments passed to f.

    Returns:
    - d: relative error between analytical and numerical gradient.
    """

    # Compute function value and analytical gradient
    fX, dfX = f(X, *args)

    # Initialize numerical gradient
    dh = np.zeros_like(X)

    for j in range(len(X)):
        dx = np.zeros_like(X)
        dx[j] = e  # Small perturbation in one dimension

        # Compute function values at perturbed points
        y2, _ = f(X + dx, *args)
        y1, _ = f(X - dx, *args)

        # Compute finite difference approximation
        dh[j] = (y2 - y1) / (2 * e)

    # Compute relative error
    error = np.linalg.norm(dh - dfX) / np.linalg.norm(dh + dfX)

    # Print comparison
    print("Analytical Gradient vs. Numerical Approximation:")
    print(np.column_stack((dfX, dh)))
    
    return error



def main():

    X_init = np.array([10.0])  # Initial guess
    X_opt, fX_vals, num_iters = minimize(X_init, example_function, 50)

    print("Optimized X:", X_opt)
    print("Function values during optimization:", fX_vals)
    print("Iterations used:", num_iters)


    X_test = np.array([2.0, 4.0, 1.5])  # Example test point
    grad_error = checkgrad(example_function, X_test)

    print(f"Gradient check error: {grad_error}")


if __name__ == "__main__":
    main()





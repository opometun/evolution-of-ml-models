"""
Task 2: Linear Regression with PyTorch [6 points]

Learning objectives:
- Get familiar with PyTorch tensors and basic linear algebra operations
- Implement the closed-form (normal equation) solution for linear regression
- Visualize data and the fitted regression line
"""

import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)


# 2.2 Closed-Form Solution [3 points]
# Implement the normal equation theta = (X^T X)^(-1) X^T y
# using PyTorch tensor operations to find the optimal parameters [w, b].

def closed_form_solution(x, y):
    """Compute optimal [slope, intercept] via the normal equation.

    Args:
        x: Input features, shape (N,)
        y: Targets, shape (N,)

    Returns:
        theta: Tensor [slope, intercept]
    """
    # TODO: Your solution here
    ones = torch.ones_like(x)
    X = torch.stack((x, ones), dim=1)

    theta = torch.linalg.inv(X.T @ X) @ X.T @ y
    return theta

# 2.3 Visualize the Fit [2 points]
# Plot the data points together with the fitted regression line.
# Also plot the true line y = 3x + 2 for comparison.

if __name__ == "__main__":
    N = 100

    # 2.1 Generate Synthetic Data [1 point]
    # Generate N = 100 data points from the model y = 3x + 2 + eps
    # where eps ~ N(0, 1) using PyTorch tensors. Plot the data.
    x = torch.linspace(0,20, N)
    eps = torch.randn(N)

    y = x * 3 + 2 + eps

    theta = closed_form_solution(x, y)
    y_fitted = theta[0] * x + theta[1]

    plt.scatter(x, y, color="green")
    plt.plot(x, 3*x+2, color="orange")
    plt.plot(x, y_fitted, color="red")
    plt.show()

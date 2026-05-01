"""
Task 3: Logistic Regression with PyTorch [8 points]

Learning objectives:
- Implement sigmoid and binary cross-entropy using PyTorch tensors
- Train a binary classifier using gradient descent
- Visualize decision boundaries
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

torch.manual_seed(42)


# 3.1 Activation Functions [1 point]
# Implement numerically stable versions of sigmoid and softmax using PyTorch.

def sigmoid(z):
    """Numerically stable sigmoid function.

    Args:
        z: Input tensor.

    Returns:
        Tensor of same shape with values in (0, 1).
    """
    return torch.where(
        z >=0,
        1 / (1 + torch.exp(-z)),
        torch.exp(z) / (1 + torch.exp(z)) 
    )


def softmax(z):
    """Numerically stable softmax. z has shape (N, K).

    Args:
        z: Input tensor, shape (N, K).

    Returns:
        Tensor of same shape, rows sum to 1.
    """
    max_z = torch.max(z, dim=1, keepdim=True).values
    z_shifted = z - max_z
    exp_z_shifted = torch.exp(z_shifted)
    sum_exp_z_shifted = torch.sum(exp_z_shifted, dim=1, keepdim=True)
    return exp_z_shifted / sum_exp_z_shifted



# 3.2 Data and Loss [2 points]
# Generate the moons dataset and implement binary cross-entropy.

def binary_cross_entropy(y_true, y_pred):
    """Compute the mean binary cross-entropy loss.

    Args:
        y_true: True labels, shape (N,), values in {0, 1}.
        y_pred: Predicted probabilities, shape (N,), values in (0, 1).

    Returns:
        Scalar loss.
    """
    term_1 = y_true * torch.log(y_pred)
    term_2 = (1 - y_true) * torch.log(1 - y_pred)
    return -torch.mean(term_1 + term_2)


# 3.3 Training [2 points]
# Implement logistic regression training with gradient descent.

def logistic_regression_train(X_train, y_train, lr=0.1, n_iterations=1000):
    """Train binary logistic regression using gradient descent.

    Args:
        X_train: Training features, shape (N, D).
        y_train: Training labels, shape (N,).
        lr: Learning rate.
        n_iterations: Number of gradient descent steps.

    Returns:
        w: Weights, shape (D,).
        b: Bias (scalar tensor).
        losses: List of loss values per iteration.
    """
    N, D = X_train.shape
    w = torch.zeros(D, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    losses = []

    for iteration in range(n_iterations):
        y_pred = sigmoid(X_train @ w + b)
        loss = binary_cross_entropy(y_train, y_pred)

        # Store only the numeric values
        losses.append(loss.item())
        
        # Compute the gradients
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()
    
    return w, b, losses


# 3.4 Evaluate and Visualize [2 points]
# Compute accuracy on train and test sets. Plot the decision boundary.

def predict(X, w, b, threshold=0.5):
    """Predict class labels.

    Args:
        X: Input features, shape (N, D).
        w: Weights, shape (D,).
        b: Bias (scalar tensor).
        threshold: Classification threshold.

    Returns:
        Predicted labels, shape (N,).
    """
    y_pred = sigmoid(X @ w + b)
    return torch.where(
        y_pred > threshold,
        1,
        0
    )    
    


def plot_decision_boundary(X, y, w, b):
    """Plot data points and the linear decision boundary.

    Args:
        X: Input features, shape (N, 2).
        y: Labels, shape (N,).
        w: Weights, shape (2,).
        b: Bias (scalar tensor).
    """
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    w_np = w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else w
    b_np = b.item() if isinstance(b, torch.Tensor) else b

    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="bwr", alpha=0.6, edgecolors='k')
    
    # Find the range of the x-axis (feature x1)
    x1_min, x1_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    x1_values = np.linspace(x1_min, x1_max, 100)
    
    # Use our formula: x2 = -(w1*x1 + b) / w2
    # w[0] is w1, w[1] is w2
    x2_values = -(w_np[0] * x1_values + b_np) / w_np[1]

    # 3. Plot the line
    plt.plot(x1_values, x2_values, color='black', lw=2, label="Decision Boundary")
    
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Generate data
    X_np, y_np = make_moons(n_samples=300, noise=0.2, random_state=42)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
    )
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    plt.scatter(X_train_np[:, 0], X_train_np[:, 1], c=y_train_np, cmap="bwr", alpha=0.6)
    plt.title("Moons Dataset (Train)")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

    # Train logistic regression and store the learned parameters.
    w, b, losses = logistic_regression_train(X_train, y_train)

    # Compute and print train/test accuracy.
    y_train_pred = predict(X_train, w, b)
    y_test_pred = predict(X_test, w, b)

    train_acc = (y_train_pred == y_train).float().mean()
    test_acc = (y_test_pred == y_test).float().mean()

    print("Train accuracy:", train_acc.item())
    print("Test accuracy:", test_acc.item())

    # Plot the decision boundary.
    plot_decision_boundary(X_train, y_train, w, b)

#3.5 Reflextion
# Because the moon dataset is not separable by the line, we need a curve for that
# We should use polynomial function instead of linear and even better to switch to more sophisticated algo

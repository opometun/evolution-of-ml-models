import matplotlib.pyplot as plt
import torch

from data_prep import load_simple_data

EPOCHS = 1000
LEARNING_RATE = 0.00001


def forward(x, w, b):
    """
    Simple linear model:
        y_hat = w * x + b
    """
    return w * x + b

def train_model(x, y):
    """
    Train a 1D linear regression model on BMI -> charges.
    """
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD([w, b], lr=LEARNING_RATE)

    loss_history = []

    for epoch in range(EPOCHS):
        predictions = w * x + b
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())

    return w, b, loss_history

def plot_results(x, y, w, b):
    """
    Plot the raw data and the final prediction line.
    """
    plt.scatter(x=x.squeeze(), y=y.squeeze(), color="green", label="Data")

    y_hat = forward(x=x, w=w, b=b).detach()
    sorted_idx = torch.argsort(x.squeeze())
    x_sorted = x[sorted_idx].squeeze()
    y_hat_sorted = y_hat[sorted_idx].squeeze()

    plt.plot(x_sorted, y_hat_sorted, label="Regression line")

    plt.title("Simple Linear Regression")
    plt.xlabel("BMI")
    plt.ylabel("Charges")
    plt.legend()
    plt.show()

def main():
    x, y = load_simple_data()

    w, b, loss_history = train_model(x, y)

    print(f"Learned parameter w: {w}")
    print(f"Learned parameter b: {b}")
    print(f"Learned final loss: {loss_history[-1]}")

    plot_results(x, y, w, b)
    


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import torch

from data_prep import load_simple_data


# TODO: choose your own training hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.00001


def forward(x, w, b):
    """
    Simple linear model:
        y_hat = w * x + b
    """

    # TODO: return the prediction formula
    raise NotImplementedError("Complete forward()")


def train_model(x, y):
    """
    Train a 1D linear regression model on BMI -> charges.
    """

    # TODO: initialize w and b as tensors
    # TODO: set requires_grad=True for both
    # w = ...
    # b = ...

    # TODO: create the loss function
    # loss_fn = torch.nn.MSELoss()

    # TODO: create the optimizer and pass [w, b]
    # optimizer = torch.optim.SGD([...], lr=LEARNING_RATE)

    # TODO: create a list to store the loss values if you want
    # loss_history = []

    # TODO: write the training loop
    # for epoch in range(EPOCHS):
    #     predictions = ...
    #     loss = ...
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     loss_history.append(loss.item())

    raise NotImplementedError("Complete train_model()")


def plot_results(x, y, w, b):
    """
    Plot the raw data and the final prediction line.
    """

    # TODO: make a scatter plot of x vs y
    # plt.scatter(...)

    # TODO: create model predictions for the line
    # y_hat = ...

    # TODO: plot the line on top of the scatter plot
    # plt.plot(...)

    # TODO: add title, axis labels, legend
    # TODO: call plt.show() or save the figure
    raise NotImplementedError("Complete plot_results()")


def main():
    # TODO: load x and y from data_prep.py
    # x, y = load_simple_data()

    # TODO: train the model
    # w, b, loss_history = train_model(x, y)

    # TODO: print the learned parameters and final loss

    # TODO: visualize the fitted line
    # plot_results(x, y, w, b)
    pass


if __name__ == "__main__":
    main()

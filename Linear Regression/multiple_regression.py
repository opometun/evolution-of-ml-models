import torch

from data_prep import load_multiple_data


# TODO: choose your own training hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.00001


def forward(X, w, b):
    """
    Multiple linear model:
        y_hat = X @ w + b
    """

    # TODO: return the matrix multiplication formula
    raise NotImplementedError("Complete forward()")


def train_model(X, y):
    """
    Train a multiple linear regression model on [bmi, age, smoker] -> charges.
    """

    # TODO: initialize w with one weight per feature
    # Hint: if X has 3 columns, w should match that feature count.
    # w = ...

    # TODO: initialize b
    # b = ...

    # TODO: set requires_grad=True for both parameters

    # TODO: create MSE loss
    # loss_fn = torch.nn.MSELoss()

    # TODO: create SGD optimizer
    # optimizer = torch.optim.SGD([...], lr=LEARNING_RATE)

    # TODO: store losses if you want to track learning
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


def main():
    # TODO: load X and y from data_prep.py
    # X, y = load_multiple_data()

    # TODO: train the model
    # w, b, loss_history = train_model(X, y)

    # TODO: print the learned weights, bias, and final loss
    pass


if __name__ == "__main__":
    main()

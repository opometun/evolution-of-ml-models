import torch
import matplotlib.pyplot as plt
from datasets import load_dataset


# TODO:
# 1. Replace the dataset name below with the one your assignment expects.
# 2. Load the dataset split you want to use, for example "train".
# 3. Inspect the available columns before extracting features.
DATASET_NAME = "pgurazada1/medical-insurance"


def load_simple_data():
    """
    Return:
        x: BMI values as a float tensor
        y: Charges values as a float tensor
    """
    ds = load_dataset(DATASET_NAME)

    split = ds["train"]

    bmi_values = split["bmi"]
    charges_values = split["charges"]

    x = torch.tensor(bmi_values, dtype=torch.float32)
    y = torch.tensor(charges_values, dtype=torch.float32)

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)

    return x, y
    #raise NotImplementedError("Complete load_simple_data()")


def load_multiple_data():
    """
    Return:
        X: 2D tensor with columns [bmi, age, smoker]
        y: charges as a float tensor
    """
    ds = load_dataset(DATASET_NAME)

    split = ds["train"]

    bmi_values = split["bmi"]
    age_values = split["age"]

    #smoker in 1.0/0.0 format
    smoker_values = list(map(lambda x: 1.0 if x == "yes" else 0.0, split["smoker"]))

    charges_values = split["charges"]

    # TODO: convert all features to float32 tensors
    bmi_tensor = torch.tensor(bmi_values, dtype=torch.float32)
    age_tensor = torch.tensor(age_values, dtype=torch.float32)
    smoker_tensor = torch.tensor(smoker_values, dtype=torch.float32)
    y = torch.tensor(charges_values, dtype=torch.float32)

    X = torch.stack((bmi_tensor, age_tensor, smoker_tensor), dim=1)
    y = y.unsqueeze(1)

    return X, y
    #raise NotImplementedError("Complete load_multiple_data()")


def preview_data():
    """
    Optional helper for checking shapes or plotting raw data.
    """

    X, y = load_multiple_data()
    
    print(f"Tensor X shape {X.shape}")
    print(f"Tensor y shape {y.shape}")

    plt.plot(X[:,2], y, "bo")
    plt.show()


if __name__ == "__main__":
    preview_data()

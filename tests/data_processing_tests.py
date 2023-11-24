import numpy as np
import pandas as pd

from utils.data_processing import train_test_split


def training_split_tests(
    df: pd.DataFrame, col: str, iterations: int, stratify_col="stroke"
):
    """
    Test the proportions of a specific column in the train and test sets, with and without stratification.

    Parameters:
    df (pandas.DataFrame): The DataFrame to sample from.
    col (str): The column to calculate proportions for.
    iterations (int): The number of iterations to perform.
    stratify_col (str): The column to use for stratification.

    Prints:
    Mean and standard deviation of the proportions in the train and test sets, with and without stratification.
    """
    results = {"train_stratify": [], "test_stratify": [], "train": [], "test": []}

    for i in range(iterations):
        for stratify in [True, False]:
            train, test = train_test_split(
                df, stratify_col, stratify=stratify, random_state=i
            )
            results["train" + ("_stratify" if stratify else "")].append(
                train[col].mean()
            )
            results["test" + ("_stratify" if stratify else "")].append(test[col].mean())

    for key, values in results.items():
        print(f"mean {key} prop: {round(np.mean(values), 6)}")
        print(f"std {key} prop: {round(np.std(values), 6)}")


if __name__ == "__main__":
    data = pd.read_csv("data/stroke.csv", delimiter=",")
    training_split_tests(data, "stroke", 1000)

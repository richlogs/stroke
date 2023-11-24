import numpy as np
import pandas as pd

from utils.data_processing import train_test_split


def train_test_split(
    df, stratify_col=None, test_size=0.2, random_state=None, stratify=True
):
    """
    Perform stratified or non-stratified random sampling on a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to sample from.
    stratify_col (str): The column to use for stratification.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.
    stratify (bool): Whether to perform stratified sampling.

    Returns:
    pandas.DataFrame, pandas.DataFrame: The training set and the test set.
    """
    if stratify and stratify_col is not None:
        test = df.groupby(stratify_col).apply(
            lambda x: x.sample(frac=test_size, random_state=random_state)
        )
        test.index = test.index.droplevel(0)
        train = df.drop(test.index)
    else:
        test = df.sample(frac=test_size, random_state=random_state)
        train = df.drop(test.index)

    train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train, test


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

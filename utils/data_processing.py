from typing import Tuple

import pandas as pd


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns of a DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame to encode.

    Returns:
    pandas.DataFrame: The encoded DataFrame.
    """
    for column in df.select_dtypes(include="object"):
        df[column] = pd.factorize(df[column])[0]
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns of a DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame to encode.

    Returns:
    pandas.DataFrame: The encoded DataFrame.
    """
    return pd.get_dummies(df, drop_first=True, dtype=int)


def train_test_split(
    df: pd.DataFrame,
    stratify_col: str | None = None,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

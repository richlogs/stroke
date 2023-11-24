import pandas as pd


def encode_categories(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.select_dtypes(include="object"):
        encoding_map = {level: i for i, level in enumerate(data[column].unique())}
        data[column] = data[column].map(encoding_map)
    return data


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

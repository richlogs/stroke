import numpy as np
import pandas as pd


def encode_categories(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.select_dtypes(include="object"):
        encoding_map = {level: i for i, level in enumerate(data[column].unique())}
        data[column] = data[column].map(encoding_map)
    return data


# def train_test_split_old(df: pd.DataFrame, outcome_var: str, train_size: float = 0.7, stratify: bool = True, random_seed: int = 42) -> pd.DataFrame:
#     np.random.seed(random_seed)  # Set random seed for reproducibility

#     if stratify:
#         def add_random_uniform(group):
#             group['random'] = np.random.uniform(size=len(group))
#             return group

#         df = df.groupby(outcome_var).apply(add_random_uniform)
#         df = df.reset_index()
#     else:
#         df["random"] = np.random.uniform(high=1, size=len(df))

#     df.sort_values(by='random', inplace=True)
#     df.drop(columns='random', inplace=True)

#     split_index = int(train_size * len(df))
#     df_train, df_test = df.iloc[:split_index], df.iloc[split_index:]

#     return df_train, df_test


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


def test_cases(df: pd.DataFrame, col: str, iterations: int):
    train_list_stratify = []
    test_list_stratify = []
    train_list = []
    test_list = []

    for i in range(iterations):
        # train_stratify, test_stratify =  train_test_split_old(df, "stroke", stratify=True, random_seed=i)
        train_stratify, test_stratify = train_test_split(df, "stroke", stratify=True)
        train_prop_stratify = train_stratify[col].sum() / len(train_stratify)
        test_prop_stratify = test_stratify[col].sum() / len(test_stratify)
        train_list_stratify.append(train_prop_stratify)
        test_list_stratify.append(test_prop_stratify)

    for i in range(iterations):
        # train, test =  train_test_split_old(df, "stroke", stratify=False, random_seed=i)
        train, test = train_test_split(df, stratify=False)
        train_prop = train[col].sum() / len(train)
        test_prop = test[col].sum() / len(test)
        train_list.append(train_prop)
        test_list.append(test_prop)

    print(f"mean train prop stratify: {round(np.mean(train_list_stratify), 6)}")
    print(f"mean train prop:          {round(np.mean(train_list), 6)}")
    print(f"mean test prop stratify:  {round(np.mean(test_list_stratify), 6)}")
    print(f"mean test prop:           {round(np.mean(test_list), 6)}")
    print(f"std train prop stratify:  {round(np.std(train_list_stratify), 6)}")
    print(f"std train prop:           {round(np.std(train_list), 6)}")
    print(f"std test prop stratify:   {round(np.std(test_list_stratify), 6)}")
    print(f"std test prop:            {round(np.std(test_list), 6)}")


if __name__ == "__main__":
    data = pd.read_csv("data/stroke.csv", delimiter=",")
    test_cases(data, "stroke", 1000)

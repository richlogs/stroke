import pandas as pd


def mode_impute(df: pd.DataFrame, col: str, value: str | None) -> pd.DataFrame:
    """
    Impute missing values in a column with the mode.

    Parameters:
    df (pandas.DataFrame): The DataFrame to impute.
    col (str): The column to impute.
    value (str): The value to replace with the mode.

    Returns:
    pandas.DataFrame: The imputed DataFrame.
    """
    mode = df[col].mode().values[0]

    if value:
        df[col].replace(value, mode, inplace=True)
    else:
        df[col].fillna(mode, inplace=True)
    return df


def mean_impute(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Impute missing values in a column with the mean.

    Parameters:
    df (pandas.DataFrame): The DataFrame to impute.
    col (str): The column to impute.

    Returns:
    pandas.DataFrame: The imputed DataFrame.
    """
    assert df[col].dtype in ["float64", "int64"], "Column must be numeric."

    mean = df[col].mean()

    if df[col].dtype == "int64":
        mean = int(mean)

    df[col].fillna(mean, inplace=True)
    return df

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

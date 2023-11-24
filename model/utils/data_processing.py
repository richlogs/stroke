import pandas as pd


def encode_categories(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.select_dtypes(include="object"):
        encoding_map = {level: i for i, level in enumerate(data[column].unique())}
        data[column] = data[column].map(encoding_map)
    return data

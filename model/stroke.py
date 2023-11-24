import pandas as pd

from utils.data_processing import encode_categories
from utils.data_processing import train_test_split


data = pd.read_csv("data/stroke.csv", delimiter=",")
processed_data = encode_categories(data)
df_train, df_test = train_test_split(data)

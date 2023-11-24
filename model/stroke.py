import pandas as pd
from utils.data_processing import encode_categories


data = pd.read_csv("data/stroke.csv", delimiter=",")
processed_data = encode_categories(data)
print(processed_data.head(10))

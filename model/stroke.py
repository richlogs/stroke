import pandas as pd

from utils.data_processing import one_hot_encode


data = pd.read_csv("data/stroke.csv", delimiter=",")
data.drop(columns=["id"], inplace=True)

one_hot = one_hot_encode(data)
x = 3
## We need to impute 201 missing values in the bmi column and 4 missing values in the smoking_status column.

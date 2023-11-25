import pandas as pd

from utils.data_processing import one_hot_encode
from utils.imputation import mode_impute


data = pd.read_csv("data/stroke.csv", delimiter=",")
data.drop(columns=["id"], inplace=True)
print(data["smoking_status"].value_counts())
data = mode_impute(data, "smoking_status", "Unknown")
print(data["smoking_status"].value_counts())

one_hot = one_hot_encode(data)
x = 3
## We need to impute 201 missing values in the bmi column and 1544 missing values in the smoking_status column.
## You may want to consider whether to one hot encode categorical columns with k or k-1 dummies, leave them as they are, numericalize them, or drop them altogether.

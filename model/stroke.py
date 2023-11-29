import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from utils.data_processing import one_hot_encode
from utils.data_processing import train_test_split
from utils.imputation import mode_impute

## Read data
data = pd.read_csv("data/stroke.csv", delimiter=",")

## Preprocess
data.drop(columns=["id"], inplace=True)
data = mode_impute(data, "smoking_status", "Unknown")
one_hot = one_hot_encode(data)

## Train test split
train, test = train_test_split(one_hot, stratify_col="stroke", stratify=True)
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

## Build Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## Accuracy
acc = sum(y_test == y_pred) / len(y_pred)
print(f"Accuracy: {round(acc * 100, 4)}%")

##
## We need to impute 201 missing values in the bmi column and 1544 missing values in the smoking_status column.
## You may want to consider whether to one hot encode categorical columns with k or k-1 dummies, leave them as they are, numericalize them, or drop them altogether.

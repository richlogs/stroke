import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from utils.data_processing import one_hot_encode
from utils.data_processing import train_test_split
from utils.imputation import mean_impute
from utils.imputation import mode_impute

seed = 50

## Read data
data = pd.read_csv("data/stroke.csv", delimiter=",")

## Preprocess
data.drop(columns=["id"], inplace=True)
data = mode_impute(data, "smoking_status", "Unknown")
data = mean_impute(data, col="bmi")
one_hot = one_hot_encode(data, drop_first=False)

## Train test split
train, test = train_test_split(
    one_hot, stratify_col="stroke", stratify=True, random_state=seed
)
X_train = train.drop("stroke", axis=1)
y_train = train["stroke"]
X_test = test.drop("stroke", axis=1)
y_test = test["stroke"]

## Build Classifier
clf = DecisionTreeClassifier(random_state=seed)
clf.fit(X_train, y_train)


## Accuracy
y_pred = clf.predict(X_test)
acc = sum(y_test == y_pred) / len(y_pred)
print(f"Accuracy DT standard: {round(acc * 100, 4)}%")

## AUCPR
y_prob = clf.predict_proba(X_test)[:, 1]
aucpr_score = average_precision_score(y_test, y_prob)
print(f"AUCPR DT standard: {round(aucpr_score * 100, 4)}%")

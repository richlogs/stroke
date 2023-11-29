import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from utils.data_processing import one_hot_encode
from utils.data_processing import train_test_split
from utils.imputation import mode_impute

seed = 50

## Read data
data = pd.read_csv("data/stroke.csv", delimiter=",")

## Preprocess
data.drop(columns=["id"], inplace=True)
data = mode_impute(data, "smoking_status", "Unknown")
one_hot = one_hot_encode(data)

## Train test split
train, test = train_test_split(
    one_hot, stratify_col="stroke", stratify=True, random_state=seed
)
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

## Build Classifier
clf = DecisionTreeClassifier(random_state=seed)
clf.fit(X_train, y_train)

# Grid search aucpr
clf_auc_cv = DecisionTreeClassifier(random_state=seed)
aucpr_scorer = make_scorer(average_precision_score, needs_proba=True)
param_grid = {"max_depth": [1, 2, 3, 4, 5, 10], "min_samples_split": [2, 3, 4, 5, 10]}
grid_search = GridSearchCV(clf_auc_cv, param_grid, scoring=aucpr_scorer, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

clf_auc = DecisionTreeClassifier(
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
)
clf_auc.fit(X_train, y_train)

## Accuracy
y_pred = clf.predict(X_test)
y_pred_auc = clf_auc.predict(X_test)
acc = sum(y_test == y_pred) / len(y_pred)
acc_auc = sum(y_test == y_pred_auc) / len(y_pred_auc)
print(f"Accuracy standard DT: {round(acc * 100, 4)}%")
print(f"Accuracy aucpr DT: {round(acc_auc * 100, 4)}%")

## AUCPR
y_prob = clf.predict_proba(X_test)[:, 1]
y_prob_auc = clf_auc.predict_proba(X_test)[:, 1]
aucpr_score = average_precision_score(y_test, y_prob)
aucpr_score_auc = average_precision_score(y_test, y_prob_auc)
print(f"AUCPR standard DT: {round(aucpr_score * 100, 4)}%")
print(f"AUCPR aucpr DT: {round(aucpr_score_auc * 100, 4)}%")

import pandas as pd
from sklearn.metrics import average_precision_score


def accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return sum(y_true == y_pred) / len(y_pred)


def aucpr(y_true: pd.Series, y_prob: pd.Series) -> float:
    return average_precision_score(y_true, y_prob)


def precision(y_true: pd.Series, y_pred: pd.Series) -> float:
    return sum((y_true == 1) & (y_pred == 1)) / sum(y_pred == 1)


def recall(y_true: pd.Series, y_pred: pd.Series) -> float:
    return sum((y_true == 1) & (y_pred == 1)) / sum(y_true == 1)


def f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)

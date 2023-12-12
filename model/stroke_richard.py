from typing import Callable

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from log.log import get_stroke_logger
from utils.data_processing import one_hot_encode
from utils.data_processing import read_data
from utils.data_processing import train_test_split
from utils.imputation import mean_impute
from utils.imputation import mode_impute
from utils.metrics import accuracy
from utils.metrics import aucpr
from utils.metrics import f1
from utils.metrics import precision
from utils.metrics import recall

# set up loggerging
logger = get_stroke_logger()

# Set random seed
seed = 420  # np.random.randint(0, 10000)

logger.info(f"random seed: {seed}")

# Read data


# Preprocess
def preprocess(
    df: pd.DataFrame,
    catagorical_imputation: Callable,
    numeric_imputation: Callable,
    encoding_method: Callable,
) -> pd.DataFrame:
    df.drop(columns=["id"], inplace=True)
    df = catagorical_imputation(df, "smoking_status", "Unknown")
    df = numeric_imputation(df, col="bmi")
    df = encoding_method(df, drop_first=True)
    return df


# Split data
def split_data(
    df: pd.DataFrame, response_var: str, stratify: bool = True, random_state: int = 42
) -> pd.DataFrame:
    train, test = train_test_split(
        df, stratify_col=response_var, stratify=stratify, random_state=random_state
    )
    X_train = train.drop(response_var, axis=1)
    y_train = train[response_var]
    X_test = test.drop(response_var, axis=1)
    y_test = test[response_var]
    return X_train, y_train, X_test, y_test


# make model methods
def make_model(model_class, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
    clf = model_class(**kwargs)
    clf.fit(X_train, y_train)
    return clf


def grid_search(
    model_class, X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict, **kwargs
):
    clf = model_class(**kwargs)
    scorer = make_scorer(
        average_precision_score, needs_proba=False
    )  # needs_proba = False optimises for precision
    grid_search = GridSearchCV(clf, param_grid, scoring=scorer, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    result_dict = {
        "accuracy": accuracy(y_test, model.predict(X_test)),
        "aucpr": aucpr(y_test, model.predict_proba(X_test)[:, 1]),
        "precision": precision(y_test, model.predict(X_test)),
        "recall": recall(y_test, model.predict(X_test)),
        "f1": f1(y_test, model.predict(X_test)),
    }
    return result_dict


def main(path: str):
    df = read_data(path)
    df = preprocess(
        df,
        catagorical_imputation=mode_impute,
        numeric_imputation=mean_impute,
        encoding_method=one_hot_encode,
    )
    X_train, y_train, X_test, y_test = split_data(
        df, response_var="stroke", random_state=seed
    )
    decison_tree = make_model(
        DecisionTreeClassifier, X_train, y_train, random_state=seed
    )
    decison_tree_search = grid_search(
        DecisionTreeClassifier,
        X_train,
        y_train,
        param_grid={
            "max_depth": [1, 2, 3, 4, 5, 10],
            "min_samples_split": [2, 3, 4, 5, 10],
        },
        random_state=seed,
    )

    decison_tree_metrics = evaluate(decison_tree, X_test, y_test)
    decison_tree_search_metrics = evaluate(decison_tree_search, X_test, y_test)

    for key, value in decison_tree_metrics.items():
        logger.info(f"Decision Tree {key}: {round(value, 4)}")

    for key, value in decison_tree_search_metrics.items():
        logger.info(f"Decision Tree Search {key}: {round(value, 4)}")


if __name__ == "__main__":
    main("data/stroke.csv")

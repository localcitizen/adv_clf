"""
Module with extra functions.
"""
from functools import reduce
import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import Pipeline

from lib.scripts.global_variables import (PREPR_FEATURE_COL, LABEL_COL)


def seed_everything(seed_value: int) -> None:
    """Seeds all values."""
    random.seed(seed_value)
    np.random.seed(seed_value)


def get_report(pipe: Pipeline, df_test: pd.DataFrame, categories_col_name: str) -> tuple:
    """Returns metrics of the trained model.

    Args:
        pipe: The trained model (weight and documents' vectors).
        df_test: Test dataframe with labels to be predicted.
        categories_col_name: Column name with categories.

    Returns:
        Tuple with model metrics.
    """
    y_test = df_test[LABEL_COL].values
    y_pred = pipe.predict(df_test[PREPR_FEATURE_COL].values.tolist())
    y_pred_proba = pipe.predict_proba(df_test[PREPR_FEATURE_COL].values.tolist())

    report = metrics.classification_report(y_test, y_pred)
    accuracy = "accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred))
    roc_auc_score = "roc_auc_score: {:0.3f}".format(metrics.roc_auc_score(y_test, y_pred_proba[:, 1]))

    roc_aucs = roc_auc_for_categories(y_test, y_pred_proba[:, 1], df_test[categories_col_name])
    res = reduce(lambda x, y: x + y, roc_aucs.values())

    roc_auc_cats = ''.join(f"{key} : {val}\n" for key, val in roc_aucs.items())

    average_roc_auc = f"Average: {res / len(roc_aucs)}"

    return report, accuracy, roc_auc_score, roc_auc_cats, average_roc_auc


def roc_auc_for_categories(y_test: np.ndarray, y_pred_proba: np.ndarray, df: pd.Series) -> dict:
    """Computes roc_auc value for each category in dataset.

    Args:
        y_test: True labels for dataset.
        y_pred_proba: Predicted values for dataset.
        df: pd.Series object containing available categories.

    Returns:
        Dictionary with roc_auc values for each available category.
    """
    roc_aucs = {}
    categories = df.unique()
    for category in categories:
        category_items_idxs = df[df == category].index
        score = metrics.roc_auc_score(y_test[category_items_idxs], y_pred_proba[category_items_idxs])
        roc_aucs[category] = score

    return roc_aucs

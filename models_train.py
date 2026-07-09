from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], Any]:
    """
    Evaluate a trained binary classification model.

    Parameters
    ----------
    model
        Trained sklearn model implementing predict_proba().
    X
        Feature matrix.
    y
        Ground truth labels.
    threshold
        Decision threshold.

    Returns
    -------
    metrics
        Dictionary containing evaluation metrics.
    normalized_cm
        Normalized confusion matrix.
    """

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "f1": round(f1_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "roc_auc": round(auc(*roc_curve(y, y_prob)[:2]), 4),
        "cm": confusion_matrix(y, y_pred).tolist(),
    }

    normalized_cm = confusion_matrix(y, y_pred, normalize="true")

    return metrics, normalized_cm


def plot_confusion_matrices(train_cm, val_cm) -> None:
    """
    Plot train and validation confusion matrices.
    """

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(train_cm, annot=True, cmap="Blues")
    plt.title("Train")

    plt.subplot(1, 2, 2)
    sns.heatmap(val_cm, annot=True, cmap="Blues")
    plt.title("Validation")

    plt.tight_layout()

def evaluate_train_validation(
    model: Any,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    threshold: float = 0.5,
    show_cm: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate model on training and validation datasets.
    """

    train_metrics, train_cm = evaluate_model(
        model,
        train_X,
        train_y,
        threshold,
    )

    val_metrics, val_cm = evaluate_model(
        model,
        val_X,
        val_y,
        threshold,
    )

    if show_cm:
        plot_confusion_matrices(train_cm, val_cm)

    return {
        "train": train_metrics,
        "val": val_metrics,
    }    


from sklearn.model_selection import RandomizedSearchCV


def tune_model(
    base_model: Any,
    param_dist: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 3,
):
    """
    Perform RandomizedSearchCV and return the best estimator.
    """

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_


def evaluate_fold(
    model_configs: Dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list:
    """
    Train and evaluate every model for one LOSO fold.
    """

    fold_results = []

    for model_name, cfg in model_configs.items():

        model, best_params = tune_model(
            cfg["model"],
            cfg["params"],
            X_train,
            y_train,
        )

        print(f"{model_name}: {best_params}")

        evaluation = evaluate_train_validation(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            show_cm=False,
        )

        fold_results.append(
            {
                "model": model_name,
                "evaluation": evaluation,
            }
        )

    return fold_results


def train_loso(
    model_configs: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    logo,
    drop_cols: list = None,
    # ID_col: str = "person_id",
    # session_col: str = "session_id",
) -> Dict[str, list]:
    """
    Perform Leave-One-Subject-Out evaluation.
    """

    scores = {
        "person_id": [],
        "model": [],
        "evaluation": [],
    }

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups)
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        person_id = groups.iloc[test_idx].iloc[0]

        # X_train = X_train.drop(columns=[ID_col, session_col])
        # X_test = X_test.drop(columns=[ID_col, session_col])
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols)

        fold_results = evaluate_fold(
            model_configs,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        for result in fold_results:
            scores["person_id"].append(person_id)
            scores["model"].append(result["model"])
            scores["evaluation"].append(result["evaluation"])

    return scores


from typing import Tuple


def scores_to_dataframe(scores: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert evaluation dictionary into train/validation DataFrames.
    """

    train_rows = []
    val_rows = []

    for evaluation, model in zip(scores["evaluation"], scores["model"]):

        train = {
            k: v
            for k, v in evaluation["train"].items()
            if k not in {"cm", "threshold"}
        }
        train["model"] = model

        val = {
            k: v
            for k, v in evaluation["val"].items()
            if k not in {"cm", "threshold"}
        }
        val["model"] = model

        train_rows.append(train)
        val_rows.append(val)

    return pd.DataFrame(train_rows), pd.DataFrame(val_rows)


def print_grouped_metrics(
    df: pd.DataFrame,
    title: str,
) -> None:
    """
    Print mean ± std metrics grouped by model.
    """

    print(f"\n--- {title} ---")

    for model_name, group in df.groupby("model"):

        print(f"\nModel: {model_name}")

        for col in group.columns:

            if col == "model":
                continue

            mean = group[col].mean()
            std = group[col].std()

            if pd.isna(std):
                std = 0.0

            print(f"{col:<10}: {mean:.4f} ± {std:.4f}")


def summarize_scores(scores: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert LOSO results into DataFrames and print summary statistics.

    Parameters
    ----------
    scores
        Output returned by `train_loso`.

    Returns
    -------
    df_train
        Training metrics for every fold.
    df_val
        Validation metrics for every fold.
    """

    df_train, df_val = scores_to_dataframe(scores)

    print_grouped_metrics(df_train, "Training Metrics")
    print_grouped_metrics(df_val, "Validation Metrics")

    return df_train, df_val           
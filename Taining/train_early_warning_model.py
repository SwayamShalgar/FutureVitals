"""Train and evaluate an XGBoost ICU early-warning model.

The target marks whether the next recorded time step contains an acute
physiological crash. With hourly input data, this is the requested one-step
proxy for the 15-30 minute warning window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


DEFAULT_DATA_PATH = Path(r"D:\hospital dataset\Dataset.csv")
DEFAULT_OUTPUT_DIR = Path(r"D:\hospital dataset")
FEATURE_COLUMNS = [
    "HR",
    "O2Sat",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "FiO2",
    "pH",
    "PaCO2",
    "Lactate",
    "Shock_Index",
    "SF_Ratio",
    "ROX_Index",
    "delta_1h_HR",
    "delta_3h_HR",
    "delta_1h_MAP",
    "delta_3h_MAP",
    "delta_1h_Shock_Index",
]
REQUIRED_COLUMNS = {
    "Patient_ID",
    "Hour",
    "HR",
    "O2Sat",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "FiO2",
    "pH",
    "PaCO2",
    "Lactate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ICU acute deterioration early-warning XGBoost model."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for the clinical confusion matrix.",
    )
    return parser.parse_args()


def validate_columns(data: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))


def engineer_features_and_target(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Hour"] = pd.to_numeric(data["Hour"], errors="coerce")
    if data["Hour"].isna().any():
        raise ValueError("Hour contains missing or non-numeric values.")

    numeric_columns = REQUIRED_COLUMNS - {"Patient_ID", "Hour"}
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values(["Patient_ID", "Hour"], kind="mergesort").reset_index(drop=True)
    patient_groups = data.groupby("Patient_ID", sort=False)

    # Carry intermittent measurements forward, using room air when FiO2 has no history.
    for column in ["FiO2", "pH", "PaCO2", "Lactate"]:
        data[column] = patient_groups[column].ffill()
    data["FiO2"] = data["FiO2"].fillna(0.21)

    data["Shock_Index"] = data["HR"] / data["SBP"].replace(0, np.nan)
    data["SF_Ratio"] = data["O2Sat"] / data["FiO2"].replace(0, np.nan)
    data["ROX_Index"] = data["SF_Ratio"] / data["Resp"].replace(0, np.nan)

    for column in ["HR", "MAP", "Shock_Index"]:
        data[f"delta_1h_{column}"] = patient_groups[column].diff(1)
    for column in ["HR", "MAP"]:
        data[f"delta_3h_{column}"] = patient_groups[column].diff(3)

    data["Event_Crash"] = (
        (data["MAP"] < 65)
        | (data["Shock_Index"] > 1.0)
        | (data["O2Sat"] < 90)
        | (data["Lactate"] >= 4.0)
    )
    data["Target_Early"] = (
        data.groupby("Patient_ID", sort=False)["Event_Crash"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )
    return data


def split_by_patient(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_ids = data["Patient_ID"].drop_duplicates()
    train_ids, validation_ids = train_test_split(
        patient_ids, test_size=0.2, shuffle=True, random_state=42
    )
    train_id_set = set(train_ids)
    validation_id_set = set(validation_ids)
    train_data = data[data["Patient_ID"].isin(train_id_set)].copy()
    validation_data = data[data["Patient_ID"].isin(validation_id_set)].copy()

    sort_columns = ["Patient_ID", "Hour"]
    train_data = train_data.sort_values(sort_columns, kind="mergesort")
    validation_data = validation_data.sort_values(sort_columns, kind="mergesort")
    return train_data, validation_data


def plot_evaluation(
    y_true: pd.Series,
    y_probs: np.ndarray,
    feature_importances: pd.Series,
    output_dir: Path,
    threshold: float,
    pr_auc: float,
    roc_auc: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_probs >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_probs)

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    axes[0, 0].plot(recall, precision, color="#0b6e69", linewidth=2)
    axes[0, 0].set_title(f"Precision-Recall Curve (PR-AUC = {pr_auc:.4f})")
    axes[0, 0].set_xlabel("Recall")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(false_positive_rate, true_positive_rate, color="#c75146", linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
    axes[0, 1].set_title(f"ROC Curve (ROC-AUC = {roc_auc:.4f})")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1.05)

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No crash", "Crash"],
        yticklabels=["No crash", "Crash"],
        ax=axes[1, 0],
    )
    axes[1, 0].set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
    axes[1, 0].set_xlabel("Predicted label")
    axes[1, 0].set_ylabel("True label")

    top_importances = feature_importances.head(10).sort_values()
    top_importances.plot.barh(ax=axes[1, 1], color="#315c72")
    axes[1, 1].set_title("Top 10 Feature Importances (gain)")
    axes[1, 1].set_xlabel("Gain")
    axes[1, 1].set_ylabel("")

    figure.savefig(output_dir / "evaluation_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    confusion_figure, confusion_axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No crash", "Crash"],
        yticklabels=["No crash", "Crash"],
        ax=confusion_axis,
    )
    confusion_axis.set_title(f"ICU Early Warning Confusion Matrix (threshold = {threshold:.2f})")
    confusion_axis.set_xlabel("Predicted label")
    confusion_axis.set_ylabel("True label")
    confusion_figure.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(confusion_figure)


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    data = pd.read_csv(args.data_path, dtype={"Patient_ID": "string"})
    validate_columns(data, REQUIRED_COLUMNS)
    data = engineer_features_and_target(data)
    train_data, validation_data = split_by_patient(data)

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["Target_Early"]
    X_val = validation_data[FEATURE_COLUMNS]
    y_val = validation_data["Target_Early"]

    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("Training split has no positive Target_Early examples.")
    scale_pos_weight = float((y_train == 0).sum() / positive_count)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=25,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_probs = model.predict_proba(X_val)[:, 1]
    if y_val.nunique() < 2:
        raise ValueError("Validation split must contain both target classes for AUC metrics.")
    pr_auc = average_precision_score(y_val, y_probs)
    roc_auc = roc_auc_score(y_val, y_probs)
    predictions = (y_probs >= args.threshold).astype(int)

    print("Classification report:")
    print(classification_report(y_val, predictions, target_names=["No crash", "Crash"], zero_division=0))
    print(f"PR-AUC (Average Precision): {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Scale positive weight: {scale_pos_weight:.6f}")
    print(f"Training patients: {train_data['Patient_ID'].nunique()}")
    print(f"Validation patients: {validation_data['Patient_ID'].nunique()}")

    booster = model.get_booster()
    gain_importance = booster.get_score(importance_type="gain")
    feature_importances = pd.Series(
        {feature: float(gain_importance.get(feature, 0.0)) for feature in FEATURE_COLUMNS}
    ).sort_values(ascending=False)
    plot_evaluation(
        y_val,
        y_probs,
        feature_importances,
        args.output_dir,
        args.threshold,
        pr_auc,
        roc_auc,
    )

    model_path = args.output_dir / "early_warning_15_30m.json"
    model.save_model(model_path)
    print(f"Saved evaluation plot: {args.output_dir / 'evaluation_metrics.png'}")
    print(f"Saved confusion matrix: {args.output_dir / 'confusion_matrix.png'}")
    print(f"Saved XGBoost model: {model_path}")


if __name__ == "__main__":
    main()

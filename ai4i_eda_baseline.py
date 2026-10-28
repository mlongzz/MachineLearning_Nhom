from pathlib import Path
import argparse
import sys
import json
import platform
from datetime import datetime, timezone
from time import perf_counter
import sklearn

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, ConfusionMatrixDisplay, confusion_matrix)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def safe_print(*values):
    rendered = []
    for value in values:
        text = str(value)
        try:
            if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
                text.encode(sys.stdout.encoding)
                rendered.append(text)
            else:
                rendered.append(text)
        except UnicodeEncodeError:
            rendered.append(text.encode("ascii", "backslashreplace").decode("ascii"))
    print(*rendered)


def main(data_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    numeric = ["Air temperature [K]", "Process temperature [K]",
               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    features = ["Type"] + numeric
    required = features + ["Machine failure"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    safe_print("Dataset size:", df.shape)
    safe_print(df.head())
    df.info()
    safe_print("Duplicate rows:", df.duplicated().sum())
    df.isna().sum().rename("missing_count").to_csv(out / "missing_values.csv")
    df.describe(include="all").T.to_csv(out / "data_summary.csv")
    counts = df["Machine failure"].value_counts().sort_index()
    pd.DataFrame({"count": counts, "percent": counts / len(df) * 100}).to_csv(out / "class_distribution.csv")
    safe_print("Machine failure distribution:\n", counts)
    if df[required].isna().any().any():
        raise ValueError("Missing data detected. Check missing_values.csv before training.")
    if set(df["Machine failure"].unique()) != {0, 1}:
        raise ValueError("Machine failure labels must be exactly 0 and 1.")

    sns.countplot(data=df, x="Machine failure")
    plt.title("Machine failure distribution")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=180)
    plt.close()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, ax in zip(numeric, axes.flat):
        sns.boxplot(data=df, x="Machine failure", y=col, ax=ax)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "feature_boxplots.png", dpi=180)
    plt.close(fig)
    sns.heatmap(df[numeric].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig(out / "correlation.png", dpi=180)
    plt.close()

    X, y = df[features].copy(), df["Machine failure"].copy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    pd.DataFrame({"row_index": list(X_train.index) + list(X_val.index) + list(X_test.index),
                  "split": ["train"] * len(X_train) + ["validation"] * len(X_val) + ["test"] * len(X_test)}).to_csv(out / "split_indices.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "data_path": str(Path(data_path).resolve()),
        "n_train": len(X_train), "n_validation": len(X_val), "n_test": len(X_test),
        "features": features, "seed": 42, "test_evaluated": False,
        "note": "Initial fixed configurations; no parameter or threshold tuning.",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    }
    for name, classifier in models.items():
        preprocess = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
        ])
        model = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]
        report = classification_report(y_val, pred, digits=4, zero_division=0)
        print(name, "\n", report)
        (out / f"{name}_report.txt").write_text(report, encoding="utf-8")
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        pd.DataFrame({"row_index": X_val.index, "actual": y_val.to_numpy(),
                      "predicted": pred, "failure_probability": prob}).to_csv(out / f"{name}_validation_predictions.csv", index=False)
        rows.append({"model": name, "parameters": str(classifier.get_params()),
                     "split": "70/15/15 stratified", "seed": 42, "evaluation_set": "validation",
                     "precision": precision_score(y_val, pred, zero_division=0),
                     "recall": recall_score(y_val, pred, zero_division=0),
                     "f1": f1_score(y_val, pred, zero_division=0),
                     "roc_auc": roc_auc_score(y_val, prob),
                     "average_precision": average_precision_score(y_val, prob),
                     "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                     "fit_seconds": fit_seconds, "notes": "Initial fixed configuration"})
        ConfusionMatrixDisplay.from_predictions(y_val, pred)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(out / f"{name}_confusion_matrix.png", dpi=180)
        plt.close()
    results = pd.DataFrame(rows)
    results.to_csv(out / "experiment_log.csv", index=False)
    metrics = ["precision", "recall", "f1", "roc_auc", "average_precision"]
    comparison = results[["model"] + metrics + ["tp", "fn", "fp", "tn"]].sort_values("f1", ascending=False)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    safe_print(comparison.to_string(index=False))
    ax = results.set_index("model")[metrics].plot.bar(figsize=(13, 6), ylim=(0, 1.05))
    ax.set_ylabel("Validation score")
    ax.set_title("AI4I model comparison — validation set")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "model_comparison.png", dpi=180)
    plt.close()
    report_lines = ["# AI4I — Kết quả so sánh mô hình", "",
        "Chia phân tầng 70/15/15, seed 42; chỉ đánh giá validation. Tập test chưa sử dụng.",
        "Các cấu hình ban đầu chưa được tối ưu. Mỗi mô hình dùng ngưỡng/quy tắc predict mặc định.", "",
        "| Mô hình | Precision | Recall | F1 | ROC-AUC | AP | Bỏ sót | Báo động giả |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison.itertuples(index=False):
        report_lines.append(f"| {row.model} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.roc_auc:.4f} | {row.average_precision:.4f} | {row.fn} | {row.fp} |")
    report_lines += ["", "AP là Average Precision, không phải tích phân hình thang của đường PR.",
        "Kết quả phụ thuộc tập validation nhỏ và dữ liệu mô phỏng; không chứng minh dự đoán sớm thời điểm hỏng.",
        "Bước tiếp theo: cross-validation trên train, thử đặc trưng mới và chọn ngưỡng trên validation; sau khi chốt mới đánh giá test."]
    (out / "NHAN_XET.md").write_text("\n".join(report_lines), encoding="utf-8")
    safe_print("Results saved to:", out.resolve())
    safe_print("The test set was not evaluated; it is kept separate for the final model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"D:\đồ án máy học\ai4i2020.csv")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "results" / "comparison_5_models"))
    args = parser.parse_args()
    main(args.data, args.output)

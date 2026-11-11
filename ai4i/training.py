"""Fit one algorithm and persist the evidence for one experiment round."""
import json
import time
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from .config import NUM, FE, GRIDS, SEED
from .storage import LOCAL_TIME
from .metrics import score
from .models import build_pipeline
from .storage import dump


def fit_experiment(dataset, run, model_name, engineered, tuned, output):
    """Select parameters on training CV, then evaluate validation once."""
    target = output / run / model_name
    target.mkdir(parents=True, exist_ok=True)
    features = dataset.features
    train = dataset.train_indices
    validation = dataset.validation_indices
    actual = dataset.target
    pipeline = build_pipeline(model_name, engineered)
    started = time.perf_counter()
    started_at = datetime.now(LOCAL_TIME).isoformat(timespec="seconds")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if tuned:
            search = GridSearchCV(
                pipeline, GRIDS[model_name],
                scoring={"f1": "f1", "recall": "recall", "precision": "precision",
                         "roc_auc": "roc_auc", "ap": "average_precision"},
                refit="f1",
                cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                n_jobs=2, return_train_score=True, error_score="raise",
            )
            search.fit(features.loc[train], actual.loc[train])
            pipeline = search.best_estimator_
            cv_results = pd.DataFrame(search.cv_results_)
            cv_results.to_csv(target / "cv_results.csv", index=False)
            best = search.best_index_
            cv_mean = float(search.cv_results_["mean_test_f1"][best])
            cv_std = float(search.cv_results_["std_test_f1"][best])
            cv_train = float(search.cv_results_["mean_train_f1"][best])
            dump(target / "search_grid.json", GRIDS[model_name])
        else:
            pipeline.fit(features.loc[train], actual.loc[train])
            cv_mean = cv_std = cv_train = None

    prediction = pipeline.predict(features.loc[validation])
    probability = pipeline.predict_proba(features.loc[validation])[:, 1]
    train_score = score(
        actual.loc[train], pipeline.predict(features.loc[train]),
        pipeline.predict_proba(features.loc[train])[:, 1],
    )
    validation_score = score(actual.loc[validation], prediction, probability)
    parameters = pipeline.named_steps["model"].get_params()
    result = {
        "run": run, "model": model_name, "started_at": started_at,
        "completed_at": datetime.now(LOCAL_TIME).isoformat(timespec="seconds"),
        "engineered_features": engineered, "tuned": tuned,
        **validation_score,
        "train_f1": train_score["f1"],
        "train_recall": train_score["recall"],
        "train_precision": train_score["precision"],
        "train_validation_f1_gap": train_score["f1"] - validation_score["f1"],
        "cv_mean_f1": cv_mean, "cv_std_f1": cv_std,
        "cv_mean_train_f1": cv_train,
        "fit_seconds": time.perf_counter() - started,
        "n_features_raw": len(NUM) + 1,
        "n_features_before_encoding": len(NUM + FE if engineered else NUM) + 1,
        "n_features_after_encoding": len(pipeline.named_steps["preprocess"].get_feature_names_out()),
        "seed": SEED, "evaluation_set": "validation",
        "parameters": json.dumps(parameters, default=str), "status": "success",
    }
    predictions = pd.DataFrame({
        "row_index": validation,
        "UDI": dataset.frame.loc[validation, "UDI"].to_numpy(),
        "actual": actual.loc[validation].to_numpy(),
        "predicted": prediction, "failure_probability": probability,
    })
    predictions["outcome"] = np.select(
        [(predictions.actual == 1) & (predictions.predicted == 1),
         (predictions.actual == 1) & (predictions.predicted == 0),
         (predictions.actual == 0) & (predictions.predicted == 1)],
        ["TP", "FN", "FP"], default="TN",
    )
    predictions.to_csv(target / "validation_predictions.csv", index=False)
    errors = predictions[predictions.actual != predictions.predicted].merge(
        dataset.frame.reset_index(names="row_index"), on="row_index",
        validate="one_to_one", suffixes=("", "_source"),
    )
    errors.to_csv(target / "misclassified_samples.csv", index=False)
    dump(target / "parameters.json", parameters)
    dump(target / "input_schema.json", {
        "columns": list(features.columns), "engineered_features": engineered,
        "engineering": "Integrated in saved pipeline; pass raw sensor columns directly.",
    })
    dump(target / "train_metrics.json", train_score)
    joblib.dump(pipeline, target / "model.joblib", compress=3)
    ConfusionMatrixDisplay.from_predictions(actual.loc[validation], prediction, labels=[0, 1])
    plt.title(f"{run} {model_name}")
    plt.tight_layout()
    plt.savefig(target / "confusion_matrix.png", dpi=180)
    plt.close()
    warning_rows = [
        {"run": run, "model": model_name, "category": w.category.__name__,
         "message": str(w.message)} for w in captured
    ]
    dump(target / "fit_warnings.json", warning_rows)
    # Write the completion marker only after all other artifacts exist.
    dump(target / "metrics.json", result)
    return result, warning_rows

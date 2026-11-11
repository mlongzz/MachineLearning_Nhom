"""Independently reload every saved pipeline and verify predictions and metrics."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .metrics import score
from .storage import dump


def verify_results(dataset, output):
    out = Path(output)
    logs = pd.read_csv(out / "experiment_log.csv")
    if len(logs) != 17 or logs[["run", "model"]].duplicated().any():
        raise AssertionError("Expected 17 distinct E00-E03 model results.")
    groups = [set(dataset.train_indices), set(dataset.validation_indices), set(dataset.test_indices)]
    assert len(set.union(*groups)) == len(dataset.frame)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    raw = dataset.features.loc[dataset.validation_indices]
    actual = dataset.target.loc[dataset.validation_indices]
    checked = []
    for record in logs.to_dict("records"):
        target = out / record["run"] / record["model"]
        pipe = joblib.load(target / "model.joblib")
        saved = pd.read_csv(target / "validation_predictions.csv")
        np.testing.assert_array_equal(saved.row_index, dataset.validation_indices)
        np.testing.assert_array_equal(saved.actual, actual)
        np.testing.assert_array_equal(pipe.predict(raw), saved.predicted)
        probability = pipe.predict_proba(raw)[:, 1]
        np.testing.assert_allclose(probability, saved.failure_probability, rtol=1e-12, atol=1e-15)
        for key, value in score(actual, saved.predicted, probability).items():
            np.testing.assert_allclose(value, record[key], rtol=1e-10, atol=1e-12)
        training = json.loads((target / "train_metrics.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(training["f1"], record["train_f1"])
        if record["tuned"]:
            cv = pd.read_csv(target / "cv_results.csv")
            np.testing.assert_allclose(cv.mean_test_f1.max(), record["cv_mean_f1"])
        checked.append(f"{record['run']}/{record['model']}")
    report = {"passed": True, "verified_models": len(checked), "models": checked,
              "checks": ["disjoint complete split", "validation row IDs and labels", "saved-model predictions from raw input",
                         "probabilities", "all validation metrics", "train F1", "best CV mean F1"],
              "test_evaluated": False}
    dump(out / "verification.json", report)
    return report

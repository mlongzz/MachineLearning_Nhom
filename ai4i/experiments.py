"""Run the complete midterm protocol without predicting or selecting using test."""
from pathlib import Path
from datetime import datetime
import platform
import traceback
import hashlib
import importlib.metadata
import pandas as pd
from .config import BASE, ROUNDS, SEED
from .data import load_dataset
from .eda import export_eda
from .storage import check_output, dump, LOCAL_TIME
from .training import fit_experiment
from .reporting import export_evidence
from .verification import verify_results
from .summary import export_summary


def run_experiments(data, output):
    output = Path(output)
    digest, signature = check_output(output, data)
    dataset = load_dataset(data)
    dataset.split_table().to_csv(output / "split_indices.csv", index=False)
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob("*.py"))}
    dump(output / "metadata.json", {
        "started_at": datetime.now(LOCAL_TIME).isoformat(timespec="seconds"),
        "data_sha256": digest, "config_signature": signature,
        "code_sha256": code_hashes, "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scikit-learn", "matplotlib", "joblib")},
        "seed": SEED, "cv": "StratifiedKFold(3, shuffle=True, random_state=42)",
        "cv_scoring": "F1 of class 1", "n_train": len(dataset.train_indices),
        "n_validation": len(dataset.validation_indices), "n_test": len(dataset.test_indices),
        "test_evaluated": False, "inference_input": "six raw input columns",
        "note": "Fresh rerun; not additional independent validation evidence.",
    })
    export_eda(dataset, output / "eda")
    schedule = [("E00", name, False, False) for name in ("Dummy", *BASE)]
    schedule.extend((run, name, engineered, tuned)
                    for run, engineered, tuned in ROUNDS for name in BASE)
    rows, errors, warning_rows = [], [], []
    for run, name, engineered, tuned in schedule:
        print(f"START {run} {name}", flush=True)
        try:
            result, captured = fit_experiment(dataset, run, name, engineered, tuned, output)
            rows.append(result)
            warning_rows.extend(captured)
            pd.DataFrame(rows).to_csv(output / "experiment_log.csv", index=False)
            print(f"DONE {run} {name} F1={result['f1']:.4f}", flush=True)
        except Exception as error:
            record = {"run": run, "model": name, "error_type": type(error).__name__,
                      "message": str(error), "traceback": traceback.format_exc()}
            errors.append(record)
            target = output / run / name
            target.mkdir(parents=True, exist_ok=True)
            dump(target / "execution_error.json", record)
            print(f"FAILED {run} {name}: {error}", flush=True)
    pd.DataFrame(errors, columns=["run", "model", "error_type", "message", "traceback"]).to_csv(output / "execution_errors.csv", index=False)
    pd.DataFrame(warning_rows, columns=["run", "model", "category", "message"]).to_csv(output / "execution_warnings.csv", index=False)
    if errors:
        raise RuntimeError(f"{len(errors)} models failed. See execution_errors.csv.")
    logs = pd.DataFrame(rows)
    logs.to_csv(output / "experiment_log.csv", index=False)
    export_evidence(dataset, logs[logs.run != "E00"], output)
    export_summary(logs, output)
    verification = verify_results(dataset, output)
    dump(output / "completion.json", {
        "completed_at": datetime.now(LOCAL_TIME).isoformat(timespec="seconds"),
        "successful_runs": len(rows), "baseline_runs": 5, "additional_runs": 12,
        "execution_error_count": len(errors), "warning_count": len(warning_rows),
        "verified_models": verification["verified_models"], "test_evaluated": False,
    })
    print(f"COMPLETE: {len(rows)} models and persisted predictions verified. Test not evaluated.", flush=True)
    return output

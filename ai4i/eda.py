"""Descriptive data checks and EDA. Keep source labels and valid extreme values."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import NUM
from .storage import dump

MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def export_eda(dataset, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = dataset.frame
    counts = frame["Machine failure"].value_counts().sort_index()
    missing = frame.isna().sum()
    quality = {"rows": len(frame), "columns": len(frame.columns),
               "missing_cells": int(missing.sum()), "duplicate_rows": int(frame.duplicated().sum()),
               "failure_count": int(counts.get(1, 0)), "failure_percent": float(counts.get(1, 0) / len(frame) * 100),
               "decision": "Keep source target and valid outliers; no missing-value imputation or SMOTE."}
    missing.rename("missing_count").to_csv(out / "missing_values.csv")
    frame.describe(include="all").T.to_csv(out / "data_summary.csv")
    pd.DataFrame({"count": counts, "percent": counts / len(frame) * 100}).to_csv(out / "class_distribution.csv")
    dataset.split_table().merge(frame[["Machine failure"]], left_on="row_index", right_index=True).groupby("split")["Machine failure"].agg(["count", "sum"]).rename(columns={"sum": "failure_count"}).to_csv(out / "split_summary.csv")
    if set(MODES).issubset(frame.columns):
        mismatch = frame["Machine failure"] != frame[MODES].max(axis=1)
        quality["mode_mismatch_count"] = int(mismatch.sum())
        frame.loc[mismatch].to_csv(out / "label_mode_mismatches.csv", index=False)
        frame[MODES].sum().rename("count").to_csv(out / "failure_mode_counts.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        frame[MODES].sum().plot.bar(ax=ax, color="#1D7D8E", rot=0)
        ax.set(title="Failure mode flags (may overlap)", ylabel="Observations")
        fig.tight_layout(); fig.savefig(out / "failure_modes.png", dpi=180); plt.close(fig)
    dump(out / "data_quality.json", quality)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, rot=0, color=["#ADBCC3", "#C85238"])
    ax.set(xlabel="Machine failure", ylabel="Observations", title="Class distribution")
    for x, count in enumerate(counts): ax.text(x, count + 80, str(count), ha="center")
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column, ax in zip(NUM, axes.flat):
        ax.boxplot([frame.loc[frame["Machine failure"] == c, column] for c in (0, 1)], tick_labels=["0", "1"])
        ax.set(title=column, xlabel="Machine failure")
    axes.flat[-1].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "feature_boxplots.png", dpi=180); plt.close(fig)
    correlation = frame[NUM].corr()
    correlation.to_csv(out / "correlation.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUM)), NUM, rotation=90)
    ax.set_yticks(range(len(NUM)), NUM)
    for i in range(len(NUM)):
        for j in range(len(NUM)): ax.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out / "correlation.png", dpi=180); plt.close(fig)

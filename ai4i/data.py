"""Validate input, split observations and build deterministic features."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import NUM, FE, SEED

@dataclass
class Dataset:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    train_indices: pd.Index
    validation_indices: pd.Index
    test_indices: pd.Index

    def split_table(self):
        parts = [(self.train_indices, "train"),
                 (self.validation_indices, "validation"),
                 (self.test_indices, "test")]
        return pd.concat([
            pd.DataFrame({"row_index": indices, "split": name})
            for indices, name in parts
        ], ignore_index=True)


def load_dataset(path):
    frame = pd.read_csv(Path(path))
    required = ["UDI", "Type", *NUM, "Machine failure"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if set(frame["Machine failure"].unique()) != {0, 1}:
        raise ValueError("Machine failure must contain both 0 and 1.")
    if not np.isfinite(frame[NUM].to_numpy(dtype=float)).all():
        raise ValueError("Numeric inputs must contain finite values.")
    if frame["UDI"].duplicated().any():
        raise ValueError("UDI must be unique to trace predictions to source rows.")
    if not set(frame["Type"].unique()).issubset({"L", "M", "H"}):
        raise ValueError("Type must contain only L, M or H.")
    features = frame[["Type", *NUM]].copy()
    target = frame["Machine failure"].copy()
    train, remainder = train_test_split(
        frame.index, test_size=0.30, stratify=target, random_state=SEED
    )
    validation, test = train_test_split(
        remainder, test_size=0.50,
        stratify=target.loc[remainder], random_state=SEED,
    )
    return Dataset(frame, features, target, train, validation, test)


def augment(features):
    """Add three physical interactions without fitting or mutating input."""
    result = features.copy()
    result[FE[0]] = result[NUM[1]] - result[NUM[0]]
    result[FE[1]] = result[NUM[3]] * result[NUM[2]] * 2 * np.pi / 60
    result[FE[2]] = result[NUM[3]] * result[NUM[4]]
    return result

"""Behavioral contracts: deterministic features, valid splits, train-only fit."""
import unittest
import tempfile
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from ai4i.config import NUM, FE
from ai4i.data import augment, load_dataset
from ai4i.models import build_pipeline
from ai4i.storage import check_output

class DataContracts(unittest.TestCase):
    def fixture(self):
        frame = pd.DataFrame({name: np.arange(100, dtype=float) + i for i, name in enumerate(NUM)})
        frame["UDI"] = np.arange(1, 101)
        frame["Type"] = ["L", "M"] * 50
        frame["Machine failure"] = [0] * 90 + [1] * 10
        return frame

    def test_features_have_units_and_do_not_mutate_input(self):
        original = self.fixture()[["Type", *NUM]]
        snapshot = original.copy(deep=True)
        transformed = augment(original)
        pd.testing.assert_frame_equal(original, snapshot)
        np.testing.assert_allclose(transformed[FE[0]], original[NUM[1]] - original[NUM[0]])
        np.testing.assert_allclose(transformed[FE[1]], original[NUM[3]] * original[NUM[2]] * 2 * np.pi / 60)
        np.testing.assert_allclose(transformed[FE[2]], original[NUM[3]] * original[NUM[4]])

    def test_split_is_disjoint_complete_and_deterministic(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "data.csv"
            self.fixture().to_csv(path, index=False)
            first, second = load_dataset(path), load_dataset(path)
            pd.testing.assert_frame_equal(first.split_table(), second.split_table())
            groups = [set(first.train_indices), set(first.validation_indices), set(first.test_indices)]
            self.assertEqual([len(g) for g in groups], [70, 15, 15])
            self.assertEqual(len(set.union(*groups)), 100)
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_invalid_target_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "bad.csv"
            frame = self.fixture()
            frame["Machine failure"] = 0
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "both 0 and 1"):
                load_dataset(path)

    def test_scaler_is_fitted_only_on_supplied_train(self):
        frame = self.fixture()
        pipeline = build_pipeline("LogisticRegression")
        train = frame.iloc[:95]
        pipeline.fit(train[["Type", *NUM]], train["Machine failure"])
        scaler = pipeline.named_steps["preprocess"].named_transformers_["num"]
        np.testing.assert_allclose(scaler.mean_, train[NUM].mean().to_numpy())
        self.assertFalse(np.allclose(scaler.mean_, frame[NUM].mean().to_numpy()))

    def test_saved_engineered_pipeline_accepts_raw_input(self):
        frame = self.fixture()
        raw = frame[["Type", *NUM]]
        pipe = build_pipeline("GradientBoosting", engineered=True)
        pipe.fit(raw, frame["Machine failure"])
        expected = pipe.predict_proba(raw)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(pipe, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(raw), expected)

    def test_existing_output_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            base = Path(folder)
            data = base / "data.csv"
            self.fixture().to_csv(data, index=False)
            output = base / "results"
            check_output(output, data)
            (output / "old.txt").write_text("old experiment", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                check_output(output, data)

if __name__ == "__main__":
    unittest.main()


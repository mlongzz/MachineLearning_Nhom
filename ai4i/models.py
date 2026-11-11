"""Construct pipelines with preprocessing fitted inside training folds."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from .config import BASE, NUM, FE
from .data import augment

ESTIMATORS = {
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}


def build_pipeline(model_name, engineered=False):
    """Accept raw sensor columns for fit, predict and persisted-model inference."""
    numeric_columns = NUM + FE if engineered else NUM
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
    ])
    estimator = (DummyClassifier(strategy="most_frequent") if model_name == "Dummy"
                 else ESTIMATORS[model_name](**BASE[model_name]))
    steps = [("features", FunctionTransformer(augment, validate=False))] if engineered else []
    return Pipeline([*steps, ("preprocess", preprocess), ("model", estimator)])

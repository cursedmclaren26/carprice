"""
modeling.py
-----------
Pipeline construction and *honest* evaluation.

Why this differs from the tutorial
---------------------------------
The original notebook trained ~1000 models with different `random_state` seeds,
scored each on the test split, and kept the seed that produced the best test R².
That is selection on the test set: the reported 0.92 is an optimistic estimate and
would not reproduce on new cars. It also happened *after* the split, so the model
choice was contaminated.

Here the honest protocol is:

    train / test split (held out once)
        -> cross-validated model selection on the TRAINING data only
        -> single final score on the untouched TEST set

Cross-validation uses `RepeatedKFold`, which gives the "try many splits and look at
the spread" information the seed loop was reaching for, without ever touching test
data. `RandomizedSearchCV` / `GridSearchCV` then tunes hyper-parameters inside the
training folds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, RepeatedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import FEATURES, TARGET

CATEGORICAL = ["name", "company", "fuel_type"]
NUMERIC = ["year", "kms_driven"]

RANDOM_STATE = 42


def build_pipeline(model=None, *, sparse_output: bool = False) -> Pipeline:
    """
    ColumnTransformer + estimator, exactly the shape the tutorial used but with the
    two bugs fixed:

    * `handle_unknown="ignore"` so the app never crashes on a car/company the
      encoder never saw during training;
    * numeric features are passed through as *numbers* (they used to arrive as
      strings from the app), with a scaler where the estimator benefits from it.
    """
    if model is None:
        model = LinearRegression()

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output),
             CATEGORICAL),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", pre), ("regressor", model)])


#: candidate estimators, ordered cheapest -> most flexible
CANDIDATES = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=10.0, random_state=None),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
}


@dataclass
class EvalResult:
    name: str
    cv_r2_mean: float
    cv_r2_std: float
    cv_mae_mean: float
    test_r2: float
    test_mae: float
    test_rmse: float
    n_train: int
    n_test: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    cv_splits: int = 5,
    cv_repeats: int = 3,
    candidates: dict | None = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """
    Compare candidate models with repeated CV on the training set, pick the winner,
    refit it on all training data, then score it once on the held-out test set.

    Returns (fitted_pipeline, results_table, metadata).
    """
    candidates = candidates or CANDIDATES
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True)

    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)
    rows: list[EvalResult] = []

    for name, factory in candidates.items():
        pipe = build_pipeline(factory())
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=("r2", "neg_mean_absolute_error"),
            n_jobs=-1, return_train_score=False,
        )
        cv_r2 = scores["test_r2"]
        cv_mae = -scores["test_neg_mean_absolute_error"]

        fitted = pipe.fit(X_train, y_train)
        test = _metrics(y_test, fitted.predict(X_test))

        rows.append(EvalResult(
            name=name,
            cv_r2_mean=float(cv_r2.mean()), cv_r2_std=float(cv_r2.std()),
            cv_mae_mean=float(cv_mae.mean()),
            test_r2=test["r2"], test_mae=test["mae"], test_rmse=test["rmse"],
            n_train=len(X_train), n_test=len(X_test),
        ))
        if verbose:
            r = rows[-1]
            print(f"{name:<20} CV R2 {r.cv_r2_mean:6.4f} +/- {r.cv_r2_std:.4f}"
                  f"   test R2 {r.test_r2:6.4f}   test MAE Rs {r.test_mae:,.0f}")

    results = pd.DataFrame([r.to_dict() for r in rows]).sort_values(
        "cv_r2_mean", ascending=False, ignore_index=True)
    best_name = results.loc[0, "name"]

    # Refit the winner on the full training split, score the untouched test set.
    best = build_pipeline(candidates[best_name]()).fit(X_train, y_train)
    test = _metrics(y_test, best.predict(X_test))

    meta = {
        "best_model": best_name,
        "cv_protocol": f"RepeatedKFold(n_splits={cv_splits}, n_repeats={cv_repeats})",
        "test_size": test_size,
        "random_state": random_state,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_metrics": test,
        "n_features_in": len(FEATURES),
        "categories": {
            "name": sorted(df["name"].unique().tolist()),
            "company": sorted(df["company"].unique().tolist()),
            "fuel_type": sorted(df["fuel_type"].unique().tolist()),
        },
        #: company -> models actually seen for it, so the web form can filter
        "names_by_company": {
            company: sorted(group["name"].unique().tolist())
            for company, group in df.groupby("company")
        },
    }
    return best, results, meta


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_model(pipeline: Pipeline, path: str | Path, meta: dict | None = None) -> Path:
    """
    Persist with joblib. A plain-Python metadata sidecar is written next to it so the
    app never has to introspect the pickle to know which categories exist.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    if meta is not None:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))
    return path


def load_model(path: str | Path) -> Pipeline:
    import joblib
    return joblib.load(path)


def load_meta(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.with_suffix(".json").read_text())


def predict_one(pipeline: Pipeline, *, name: str, company: str, year: int,
                kms_driven: int, fuel_type: str) -> float:
    """
    Predict a single car. Builds a *typed* one-row frame — the tutorial's
    `np.array([...]).reshape(1, 5)` made every column a string, which silently
    worked only because OneHotEncoder was the sole consumer.
    """
    row = pd.DataFrame(
        [{"name": name, "company": company, "year": int(year),
          "kms_driven": int(kms_driven), "fuel_type": fuel_type}],
        columns=FEATURES,
    )
    return float(pipeline.predict(row)[0])

"""
market.py
---------
Preparation and training for the *market* model: the learned half of the valuation.

Input is the real CarDekho listing dump (8,148 rows, 1992-2023). Unlike the tutorial's
model, this one is not asked to memorise car names. It learns the relationship between
price and the things that actually transfer to an unseen car:

    brand, segment, age, km driven, km/year, fuel, transmission, owner,
    engine displacement, seats, service history

`model` is kept as a feature, but with unseen names folded into `other` so it cannot
become a 690-column crutch. Unknown cars are then priced from *specification and
segment*, which is what a human valuer does.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             r2_score, root_mean_squared_error)
from sklearn.model_selection import (GridSearchCV, GroupKFold, RandomizedSearchCV,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .catalogue import CATALOGUE, ModelSpec, brand_of, resolve_model
from .valuation import (BINARY_FEATURES, CATEGORICAL_FEATURES, MARKET_FEATURES,
                        NUMERIC_FEATURES, VALUATION_YEAR, _guess_segment)
from .valuation import CarQuery

RANDOM_STATE = 42

#: rows kept per (brand, model) group; everything rarer collapses into 'other'
MIN_MODEL_ROWS = 4


# --------------------------------------------------------------------------- #
# load + prepare
# --------------------------------------------------------------------------- #
def load_market_raw(path: str | Path) -> pd.DataFrame:
    """Read the CarDekho dump and normalise column names."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {"Name": "name", "name": "name", "year": "year",
              "selling_price": "price", "km_driven": "kms_driven",
              "fuel": "fuel", "seller_type": "seller_type",
              "transmission": "transmission", "owner": "owner",
              "mileage": "mileage", "engine": "engine", "max_power": "max_power",
              "torque": "torque", "seats": "seats"}
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
    if "name" not in df.columns:
        raise ValueError(f"{path}: no name column. Found {list(df.columns)}")
    for col in ("price", "year", "kms_driven", "fuel", "transmission", "owner"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column '{col}'")
    return df


def _to_number(value) -> float:
    """'1197 CC' -> 1197.0 ; '1,000' -> 1000.0 ; '82.0 bhp' -> 82.0 ; bad -> NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else np.nan


def prepare_market(df: pd.DataFrame, *, min_model_rows: int = MIN_MODEL_ROWS,
                   price_ceiling: float = 2.5e7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn raw listings into model-ready features.

    Returns (features_df, audit_df) where audit_df records what each step removed.
    """
    audit: list[tuple[str, int]] = []
    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    audit.append(("raw rows", len(df)))

    before = len(df)
    df = df.dropna(subset=["price", "year", "kms_driven"])
    df = df[(df["price"] > 20_000) & (df["price"] <= price_ceiling)]
    df = df[(df["year"] >= 1990) & (df["year"] <= VALUATION_YEAR)]
    audit.append(("plausible price/year/kms", len(df)))

    before = len(df)
    df = df.drop_duplicates(subset=["name", "year", "kms_driven", "price"])
    audit.append(("duplicate listings removed", len(df)))

    # ---- brand / model / spec from the catalogue ----
    df["brand"] = df["name"].map(brand_of).fillna("Other")
    df["fuel"] = df["fuel"].astype(str).str.strip().str.title()
    df["transmission"] = df["transmission"].astype(str).str.strip().str.title()
    df["owner"] = df["owner"].astype(str).str.strip()

    specs, models, segments = [], [], []
    for name, brand in zip(df["name"], df["brand"]):
        hit = resolve_model(brand, name)
        if hit is None:
            hit = resolve_model(None, name)
        if hit is None:
            specs.append(None)
            models.append("other")
            segments.append(_guess_segment(name, CarQuery(brand=brand, model=name)))
        else:
            model_name, spec = hit
            specs.append(spec)
            models.append(model_name)
            segments.append(spec.segment)

    df["model"] = models
    df["segment"] = segments
    df["cat_ex_showroom_lakh"] = [s.ex_showroom_lakh if s else np.nan for s in specs]
    df["cat_engine_cc"] = [s.engine_cc if s else np.nan for s in specs]
    df["cat_seats"] = [s.seats if s else np.nan for s in specs]

    # engine / seats: prefer what the listing states, fall back to the catalogue
    df["engine_cc"] = df["engine"].map(_to_number) if "engine" in df else np.nan
    df["engine_cc"] = df["engine_cc"].fillna(df["cat_engine_cc"])
    df["seats"] = pd.to_numeric(df.get("seats"), errors="coerce")
    df["seats"] = df["seats"].fillna(df["cat_seats"])
    df["has_service_history"] = 1          # not present in this source; documented

    df["age"] = (VALUATION_YEAR - df["year"]).clip(lower=0)
    df["km_per_year"] = df["kms_driven"] / df["age"].clip(lower=1)

    # ---- keep only model names with enough rows ----
    counts = df["model"].value_counts()
    kept = set(counts[counts >= min_model_rows].index)
    before_models = df["model"].nunique()
    df["model"] = df["model"].where(df["model"].isin(kept), "other")
    audit.append((f"model names pooled (kept >= {min_model_rows} rows, "
                  f"{before_models} -> {df['model'].nunique()})", len(df)))

    audit.append(("catalogue match (has segment + ex-showroom)", len(df)))
    features = df[MARKET_FEATURES + ["price", "name", "year", "cat_ex_showroom_lakh"]].copy()
    return features, pd.DataFrame(audit, columns=["step", "rows"])


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_market_pipeline(model: str = "hist_gb", **kwargs) -> Pipeline:
    """
    Preprocessor + estimator.

    Supports 4 model families:
      * 'ridge': regularized linear regression (L2)
      * 'hist_gb': histogram-based gradient boosting regressor
      * 'random_forest': random forest ensemble
      * 'gradient_boosting': gradient boosting trees

    `OneHotEncoder(handle_unknown="ignore")` is essential: the web app can submit a brand
    or model that never appeared in training, and it must degrade, not raise.
    """
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  min_frequency=3),
             CATEGORICAL_FEATURES),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )
    if model == "ridge":
        defaults = {"alpha": 3.0}
        defaults.update(kwargs)
        est = Ridge(**defaults)
    elif model == "hist_gb":
        defaults = {
            "max_iter": 500, "learning_rate": 0.06, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 1.0,
            "random_state": RANDOM_STATE
        }
        defaults.update(kwargs)
        est = HistGradientBoostingRegressor(**defaults)
    elif model == "random_forest":
        defaults = {
            "n_estimators": 150, "max_depth": 20, "min_samples_leaf": 1,
            "random_state": RANDOM_STATE, "n_jobs": -1
        }
        defaults.update(kwargs)
        est = RandomForestRegressor(**defaults)
    elif model == "gradient_boosting":
        defaults = {
            "n_estimators": 150, "learning_rate": 0.06, "max_depth": 5,
            "min_samples_leaf": 10, "random_state": RANDOM_STATE
        }
        defaults.update(kwargs)
        est = GradientBoostingRegressor(**defaults)
    else:
        raise ValueError(f"unknown market model '{model}' (choose from: ridge, hist_gb, random_forest, gradient_boosting)")
    return Pipeline([("preprocessor", pre), ("regressor", est)])


def tune_market(features: pd.DataFrame, *, test_size: float = 0.2,
                cv_splits: int = 5, n_iter_search: int = 12,
                models: tuple[str, ...] = ("ridge", "hist_gb", "random_forest", "gradient_boosting"),
                verbose: bool = True):
    """
    Run systematic hyperparameter tuning across 4 models with performance comparison.

    Cross-validation is strictly grouped by car model (`GroupKFold`) so that a
    fold never validates on near-identical variants of a car it just trained on.

    Models & Search Strategies:
      - Ridge: GridSearchCV over regularisation penalty alpha
      - HistGradientBoosting: RandomizedSearchCV over learning rate, max iterations,
        max leaf nodes, min samples per leaf, and L2 regularization
      - RandomForest: RandomizedSearchCV over number of trees, max depth,
        split / leaf constraints, and feature subsampling
      - GradientBoosting: RandomizedSearchCV over learning rate, tree count,
        depth, leaf constraints, and sample fractions

    Returns (best_pipeline, meta_dict, (X_test, y_test, best_pred), comparison_df)
    """
    import time

    X = features[MARKET_FEATURES]
    y = features["price"].astype(float)
    groups = features["model"]

    X_train, X_test, y_train, y_test, g_train, _ = train_test_split(
        X, y, groups, test_size=test_size, random_state=RANDOM_STATE, shuffle=True)

    gkf = GroupKFold(n_splits=cv_splits)

    param_spaces = {
        "ridge": {
            "search": GridSearchCV,
            "estimator": build_market_pipeline("ridge"),
            "params": {
                "regressor__alpha": [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            },
            "search_kwargs": {}
        },
        "hist_gb": {
            "search": RandomizedSearchCV,
            "estimator": build_market_pipeline("hist_gb"),
            "params": {
                "regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "regressor__max_iter": [200, 350, 500],
                "regressor__max_leaf_nodes": [21, 31, 45, 63],
                "regressor__min_samples_leaf": [10, 20, 30],
                "regressor__l2_regularization": [0.0, 0.5, 1.0, 3.0],
            },
            "search_kwargs": {"n_iter": n_iter_search, "random_state": RANDOM_STATE}
        },
        "random_forest": {
            "search": RandomizedSearchCV,
            "estimator": build_market_pipeline("random_forest"),
            "params": {
                "regressor__n_estimators": [100, 150, 200],
                "regressor__max_depth": [15, 22, 30, None],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
                "regressor__max_features": [1.0, "sqrt", 0.8],
            },
            "search_kwargs": {"n_iter": n_iter_search, "random_state": RANDOM_STATE}
        },
        "gradient_boosting": {
            "search": RandomizedSearchCV,
            "estimator": build_market_pipeline("gradient_boosting"),
            "params": {
                "regressor__n_estimators": [100, 150, 200],
                "regressor__learning_rate": [0.04, 0.08, 0.12],
                "regressor__max_depth": [3, 4, 5],
                "regressor__min_samples_leaf": [5, 10, 20],
                "regressor__subsample": [0.8, 0.9, 1.0],
            },
            "search_kwargs": {"n_iter": n_iter_search, "random_state": RANDOM_STATE}
        }
    }

    results = {}
    fitted_pipes = {}
    test_preds = {}

    if verbose:
        print(f"\nHyperparameter tuning across {len(models)} models with {cv_splits}-fold GroupKFold...")

    for kind in models:
        if kind not in param_spaces:
            raise ValueError(f"unknown model '{kind}' for tuning")
        cfg = param_spaces[kind]
        t0 = time.time()
        search = cfg["search"](
            cfg["estimator"],
            cfg["params"],
            cv=gkf,
            scoring="r2",
            n_jobs=-1,
            refit=True,
            **cfg["search_kwargs"]
        )
        search.fit(X_train, y_train, groups=g_train)
        best_pipe = search.best_estimator_
        pred = best_pipe.predict(X_test)
        dt = time.time() - t0

        fitted_pipes[kind] = best_pipe
        test_preds[kind] = pred

        cv_r2 = float(search.best_score_)
        test_r2 = float(r2_score(y_test, pred))
        test_mae = float(mean_absolute_error(y_test, pred))
        test_rmse = float(root_mean_squared_error(y_test, pred))
        test_mape = float(mean_absolute_percentage_error(y_test, pred))
        best_params = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}

        results[kind] = {
            "best_params": best_params,
            "cv_r2": cv_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "fit_time_sec": round(dt, 1)
        }

        if verbose:
            print(f"  {kind:<18} Grouped-CV R2: {cv_r2:.3f} | Test R2: {test_r2:.3f} | "
                  f"Test MAE: Rs {test_mae:,.0f} | Test MAPE: {test_mape:.1%} | ({dt:.1f}s)")
            print(f"    best params: {best_params}")

    winner = max(results, key=lambda k: results[k]["cv_r2"])
    best = fitted_pipes[winner]
    best_pred = test_preds[winner]

    comparison_rows = []
    for name, m in results.items():
        row = {
            "model": name,
            "cv_r2": m["cv_r2"],
            "test_r2": m["test_r2"],
            "test_mae": m["test_mae"],
            "test_rmse": m["test_rmse"],
            "test_mape": m["test_mape"],
            "fit_time_sec": m["fit_time_sec"],
            "best_params": m["best_params"],
        }
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows).sort_values("cv_r2", ascending=False)

    meta = {
        "market_model": winner,
        "cv": f"GroupKFold({cv_splits}) grouped by model name (Tuned)",
        "tuned": True,
        "results": results,
        "test_metrics": results[winner],
        "comparison": comparison_rows,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "rows": int(len(features)),
        "year_min": int(features["year"].min()) if "year" in features else None,
        "year_max": int(features["year"].max()) if "year" in features else None,
        "brands": sorted(features["brand"].unique().tolist()),
        "models_known": sorted(features.loc[features["model"] != "other", "model"]
                               .unique().tolist()),
    }

    if verbose:
        print(f"\nWinner: {winner} (Grouped-CV R2 = {results[winner]['cv_r2']:.4f})")

    return best, meta, (X_test, y_test, best_pred), comparison_df


def train_market(features: pd.DataFrame, *, test_size: float = 0.2,
                 cv_splits: int = 5, tune: bool = False,
                 models: tuple[str, ...] = ("ridge", "hist_gb", "random_forest", "gradient_boosting"),
                 verbose: bool = True):
    """
    Fit the market model and report honest metrics.

    If tune=True, runs hyperparameter tuning across the specified models.
    If tune=False, evaluates all specified models with calibrated defaults using GroupKFold.
    """
    if tune:
        best, meta, preds, _ = tune_market(features, test_size=test_size,
                                           cv_splits=cv_splits, models=models,
                                           verbose=verbose)
        return best, meta, preds

    X = features[MARKET_FEATURES]
    y = features["price"].astype(float)
    groups = features["model"]

    X_train, X_test, y_train, y_test, g_train, _ = train_test_split(
        X, y, groups, test_size=test_size, random_state=RANDOM_STATE, shuffle=True)

    results = {}
    for kind in models:
        pipe = build_market_pipeline(kind)
        gkf = GroupKFold(n_splits=cv_splits)
        cv = cross_validate(pipe, X_train, y_train, cv=gkf, groups=g_train,
                            scoring=("r2", "neg_mean_absolute_error"), n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[kind] = {
            "cv_r2": float(cv["test_r2"].mean()),
            "cv_r2_std": float(cv["test_r2"].std()),
            "test_r2": float(r2_score(y_test, pred)),
            "test_mae": float(mean_absolute_error(y_test, pred)),
            "test_rmse": float(root_mean_squared_error(y_test, pred)),
            "test_mape": float(mean_absolute_percentage_error(y_test, pred)),
        }
        if verbose:
            r = results[kind]
            print(f"{kind:<18} grouped-CV R2 {r['cv_r2']:.3f} +/- {r['cv_r2_std']:.3f}"
                  f"   test R2 {r['test_r2']:.3f}   MAE Rs {r['test_mae']:,.0f}"
                  f"   MAPE {r['test_mape']:.1%}")

    winner = max(results, key=lambda k: results[k]["cv_r2"])
    best = build_market_pipeline(winner).fit(X_train, y_train)
    pred = best.predict(X_test)

    meta = {
        "market_model": winner,
        "cv": f"GroupKFold({cv_splits}) grouped by model name",
        "tuned": False,
        "results": results,
        "test_metrics": results[winner],
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "rows": int(len(features)),
        "year_min": int(features["year"].min()) if "year" in features else None,
        "year_max": int(features["year"].max()) if "year" in features else None,
        "brands": sorted(features["brand"].unique().tolist()),
        "models_known": sorted(features.loc[features["model"] != "other", "model"]
                               .unique().tolist()),
    }
    if verbose:
        print(f"\nwinner: {winner}")
    return best, meta, (X_test, y_test, pred)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_market(pipeline, meta: dict, path: str | Path,
                comparison_df: pd.DataFrame | None = None) -> Path:
    import joblib, json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    if comparison_df is not None:
        comp_csv = path.parent / "model_comparison.csv"
        comp_json = path.parent / "model_comparison.json"
        comparison_df.to_csv(comp_csv, index=False)
        comp_json.write_text(json.dumps(meta.get("comparison", []), indent=2))
    return path


def load_market(path: str | Path):
    import joblib, json
    path = Path(path)
    model = joblib.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return model, meta

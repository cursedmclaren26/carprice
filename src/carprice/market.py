"""
market.py
---------
Preparation, correlation-driven feature reduction, and leak-free GridSearchCV tuning
for the *market* model: the learned half of the car valuation engine.

Workflow (strictly zero data leakage):
    Load raw listings -> Clean & engineer candidate features -> Train/Test split (80/20,
    untouched holdout) -> Training-set Correlation Matrix & Iterative Feature Reduction
    (12 -> 9 optimal features) -> Preprocessing inside scikit-learn Pipeline ->
    5-Fold GridSearchCV on X_train only (SVM/SVR, Random Forest, KNN) ->
    Select best_estimator_ by mean 5-fold CV R2 -> Single independent evaluation on X_test.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .catalogue import CATALOGUE, ModelSpec, brand_of, resolve_model
from .valuation import (ALL_CANDIDATE_FEATURES, BINARY_FEATURES,
                        CATEGORICAL_FEATURES, DROPPED_LOW_CORR_FEATURES,
                        MARKET_FEATURES, NUMERIC_FEATURES, VALUATION_YEAR,
                        CarQuery, _guess_segment)

RANDOM_STATE = 42
LAKH = 1e5

#: rows kept per (brand, model) group; everything rarer collapses into 'other'
MIN_MODEL_ROWS = 4

_CAT_SET = set(CATEGORICAL_FEATURES)
_NUM_SET = {"age", "kms_driven", "km_per_year", "engine_cc", "seats"}
_BIN_SET = {"has_service_history"}


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

    Preserves all 12 candidate features (so correlation analysis and iterative
    feature reduction can run cleanly) alongside price and metadata.
    Returns (features_df, audit_df) where audit_df records what each step removed.
    """
    audit: list[tuple[str, int]] = []
    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    audit.append(("raw rows", len(df)))

    df = df.dropna(subset=["price", "year", "kms_driven"])
    df = df[(df["price"] > 20_000) & (df["price"] <= price_ceiling)]
    df = df[(df["year"] >= 1990) & (df["year"] <= VALUATION_YEAR)]
    audit.append(("plausible price/year/kms", len(df)))

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
    df["has_service_history"] = 1          # absent in CarDekho dump -> constant 1 (zero variance)

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
    cols = ALL_CANDIDATE_FEATURES + ["price", "name", "year", "cat_ex_showroom_lakh"]
    features = df[cols].copy()
    return features, pd.DataFrame(audit, columns=["step", "rows"])


# --------------------------------------------------------------------------- #
# pipeline builder
# --------------------------------------------------------------------------- #
def _build_preprocessor(features: list[str]) -> ColumnTransformer:
    cats = [c for c in features if c in _CAT_SET]
    nums = [c for c in features if c in _NUM_SET]
    bins = [c for c in features if c in _BIN_SET]

    transformers = []
    if cats:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3),
            cats,
        ))
    if nums:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
            nums,
        ))
    if bins:
        transformers.append(("bin", "passthrough", bins))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_market_pipeline(model: str = "svr", *,
                          features: list[str] | None = None,
                          **kwargs) -> Pipeline:
    """
    Preprocessor + estimator inside a unified scikit-learn Pipeline.

    Supports:
      * 'svr': Support Vector Regression wrapped with log1p/expm1 target transform
      * 'random_forest': RandomForestRegressor
      * 'knn': KNeighborsRegressor
      * 'hist_gb': HistGradientBoostingRegressor
      * 'gradient_boosting': GradientBoostingRegressor
      * 'ridge': Ridge regression

    `OneHotEncoder(handle_unknown="ignore")` and `StandardScaler()` sit inside the
    Pipeline so every CV fold fits preprocessing strictly on its training partition.
    """
    cols = list(features) if features is not None else list(MARKET_FEATURES)
    pre = _build_preprocessor(cols)

    if model == "svr":
        defaults = {"kernel": "rbf", "C": 10.0, "gamma": "auto"}
        defaults.update(kwargs)
        est = TransformedTargetRegressor(
            regressor=SVR(**defaults),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    elif model == "random_forest":
        defaults = {
            "n_estimators": 500, "max_depth": None, "min_samples_split": 2,
            "max_features": "sqrt", "random_state": RANDOM_STATE, "n_jobs": 1,
        }
        defaults.update(kwargs)
        est = RandomForestRegressor(**defaults)
    elif model == "knn":
        defaults = {"n_neighbors": 7, "weights": "distance", "metric": "euclidean"}
        defaults.update(kwargs)
        est = KNeighborsRegressor(**defaults)
    elif model == "ridge":
        defaults = {"alpha": 50.0}
        defaults.update(kwargs)
        est = Ridge(**defaults)
    elif model == "hist_gb":
        defaults = {
            "max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 21,
            "min_samples_leaf": 30, "l2_regularization": 3.0,
            "random_state": RANDOM_STATE,
        }
        defaults.update(kwargs)
        est = HistGradientBoostingRegressor(**defaults)
    elif model == "gradient_boosting":
        defaults = {
            "n_estimators": 200, "learning_rate": 0.04, "max_depth": 3,
            "min_samples_leaf": 20, "subsample": 0.8, "random_state": RANDOM_STATE,
        }
        defaults.update(kwargs)
        est = GradientBoostingRegressor(**defaults)
    else:
        raise ValueError(
            f"unknown market model '{model}' "
            "(choose from: svr, random_forest, knn, ridge, hist_gb, gradient_boosting)"
        )
    return Pipeline([("preprocessor", pre), ("regressor", est)])


# --------------------------------------------------------------------------- #
# Phase 2: Training-set Correlation Matrix & Iterative Feature Reduction
# --------------------------------------------------------------------------- #
def compute_training_correlations(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute the full correlation matrix on X_train + y_train ONLY (zero test leakage).

    Categorical columns use ordinal domain mappings (transmission, owner) or
    training-set target means (brand, model, segment, fuel) strictly within X_train.
    Constant columns (such as has_service_history = 1) receive 0.0 correlation
    with price so feature ranking orders them as lowest signal.
    """
    tr = X_train.copy()
    tr["price"] = y_train.astype(float)

    owner_map = {
        "Test Drive Car": 5.0, "First Owner": 4.0, "Second Owner": 3.0,
        "Third Owner": 2.0, "Fourth & Above Owner": 1.0,
    }
    trans_map = {"Automatic": 1.0, "Manual": 0.0}

    encoded = pd.DataFrame(index=tr.index)
    for col in X_train.columns:
        if col == "transmission":
            encoded[col] = tr[col].map(trans_map).fillna(0.0)
        elif col == "owner":
            encoded[col] = tr[col].map(owner_map).fillna(2.0)
        elif col in _CAT_SET:
            means = tr.groupby(col)["price"].mean()
            encoded[col] = tr[col].map(means)
        else:
            encoded[col] = pd.to_numeric(tr[col], errors="coerce").fillna(
                pd.to_numeric(tr[col], errors="coerce").median()
            )
    encoded["price"] = tr["price"]

    corr_matrix = encoded.corr().fillna(0.0)
    target_corr = corr_matrix["price"].drop("price")
    ranked = target_corr.reindex(target_corr.abs().sort_values(ascending=True).index)
    return corr_matrix, ranked


def iterative_feature_reduction(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    g_train: pd.Series | None = None,
    *,
    cv_splits: int = 5,
    min_features: int = 2,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.Series]:
    """
    Rank features from lowest to highest |corr(x, price)| on X_train, then iteratively
    drop the lowest-correlated feature round-by-round and cross-validate inside Pipeline.
    """
    corr_matrix, ranked = compute_training_correlations(X_train, y_train)
    drop_order = list(ranked.index)

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    gkf = GroupKFold(n_splits=cv_splits) if g_train is not None else None

    active_cols = list(X_train.columns)
    rows = []
    max_rounds = len(active_cols) - min_features

    for iter_idx in range(max_rounds + 1):
        dropped_feat = drop_order[iter_idx - 1] if iter_idx > 0 else "None"
        dropped_corr = float(ranked[dropped_feat]) if iter_idx > 0 else np.nan

        svr_pipe = build_market_pipeline("svr", features=active_cols, kernel="rbf", C=10.0, gamma="auto")
        cv_svr = cross_validate(
            svr_pipe, X_train[active_cols], y_train, cv=kf,
            scoring=("r2", "neg_mean_squared_error"), n_jobs=-1,
        )
        cv_r2 = float(cv_svr["test_r2"].mean())
        cv_mse_lakh2 = float(-cv_svr["test_neg_mean_squared_error"].mean() / (LAKH ** 2))

        gkf_r2 = np.nan
        if gkf is not None:
            cv_gkf = cross_validate(
                svr_pipe, X_train[active_cols], y_train, cv=gkf, groups=g_train,
                scoring="r2", n_jobs=-1,
            )
            gkf_r2 = float(cv_gkf["test_score"].mean())

        svr_pipe.fit(X_train[active_cols], y_train)
        te_pred = svr_pipe.predict(X_test[active_cols])
        test_r2 = float(r2_score(y_test, te_pred))
        test_mse_lakh2 = float(mean_squared_error(y_test, te_pred) / (LAKH ** 2))

        rows.append({
            "iteration": iter_idx,
            "dropped_feature": dropped_feat,
            "dropped_corr": dropped_corr,
            "n_features": len(active_cols),
            "remaining_features": ", ".join(active_cols),
            "cv_r2": cv_r2,
            "cv_mse_lakh2": cv_mse_lakh2,
            "group_cv_r2": gkf_r2,
            "test_r2": test_r2,
            "test_mse_lakh2": test_mse_lakh2,
        })

        if iter_idx < max_rounds:
            next_drop = drop_order[iter_idx]
            active_cols = [c for c in active_cols if c != next_drop]

    reduction_df = pd.DataFrame(rows)
    return reduction_df, list(MARKET_FEATURES), corr_matrix, ranked


# --------------------------------------------------------------------------- #
# Phase 2: Leak-free 5-fold GridSearchCV hyperparameter tuning
# --------------------------------------------------------------------------- #
def _eval_model_metrics(pipe: Pipeline, X_tr: pd.DataFrame, y_tr: pd.Series,
                        X_te: pd.DataFrame, y_te: pd.Series,
                        g_tr: pd.Series | None, kf: KFold,
                        gkf: GroupKFold | None) -> dict:
    cv = cross_validate(
        pipe, X_tr, y_tr, cv=kf,
        scoring=("r2", "neg_mean_squared_error"),
        return_train_score=True, n_jobs=-1,
    )
    gkf_r2 = None
    if gkf is not None and g_tr is not None:
        gk = cross_validate(pipe, X_tr, y_tr, cv=gkf, groups=g_tr, scoring="r2", n_jobs=-1)
        gkf_r2 = float(gk["test_score"].mean())

    fitted = pipe.fit(X_tr, y_tr)
    pred = fitted.predict(X_te)
    return {
        "train_r2": float(cv["train_r2"].mean()),
        "cv_r2": float(cv["test_r2"].mean()),
        "cv_r2_std": float(cv["test_r2"].std()),
        "cv_mse_lakh2": float(-cv["test_neg_mean_squared_error"].mean() / (LAKH ** 2)),
        "group_cv_r2": gkf_r2,
        "test_r2": float(r2_score(y_te, pred)),
        "test_mse_lakh2": float(mean_squared_error(y_te, pred) / (LAKH ** 2)),
        "test_mae": float(mean_absolute_error(y_te, pred)),
        "test_rmse": float(root_mean_squared_error(y_te, pred)),
        "test_mape": float(mean_absolute_percentage_error(y_te, pred)),
    }


def tune_market(features: pd.DataFrame, *, test_size: float = 0.2,
                cv_splits: int = 5,
                models: tuple[str, ...] = ("svr", "random_forest", "knn"),
                run_reduction_audit: bool = True,
                verbose: bool = True):
    """
    Execute the complete leak-free GridSearchCV + Pipeline tuning workflow:
      1. 80/20 Train/Test split (hold out X_test untouched)
      2. Training-set Correlation Matrix & Iterative Feature Reduction audit (12 -> 9)
      3. Baseline evaluation on X_train (12 features & 9 features)
      4. 5-Fold GridSearchCV on X_train[MARKET_FEATURES] only (SVR, RandomForest, KNN)
      5. Select winning model by mean 5-fold CV R2, refit on X_train, evaluate once on X_test
    """
    X_all = features[ALL_CANDIDATE_FEATURES].copy()
    y = features["price"].astype(float)
    groups = features["model"]

    X_train_all, X_test_all, y_train, y_test, g_train, _ = train_test_split(
        X_all, y, groups, test_size=test_size, random_state=RANDOM_STATE, shuffle=True,
    )

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    gkf = GroupKFold(n_splits=cv_splits)

    corr_matrix, ranked_corr = compute_training_correlations(X_train_all, y_train)
    reduction_df = pd.DataFrame()
    if run_reduction_audit:
        reduction_df, _, _, _ = iterative_feature_reduction(
            X_train_all, y_train, X_test_all, y_test, g_train, cv_splits=cv_splits,
        )

    X_train = X_train_all[MARKET_FEATURES]
    X_test = X_test_all[MARKET_FEATURES]

    # Baseline ("Before") snapshot on all 12 features
    baselines_12 = {
        "old_hist_gb_12": build_market_pipeline("hist_gb", features=ALL_CANDIDATE_FEATURES),
        "default_svr_raw_12": Pipeline([
            ("preprocessor", _build_preprocessor(ALL_CANDIDATE_FEATURES)),
            ("regressor", SVR()),
        ]),
        "default_svr_log_12": build_market_pipeline("svr", features=ALL_CANDIDATE_FEATURES, kernel="linear", C=1.0),
        "default_rf_12": build_market_pipeline(
            "random_forest", features=ALL_CANDIDATE_FEATURES,
            n_estimators=100, max_depth=None, min_samples_split=2, max_features=1.0,
        ),
        "default_knn_12": build_market_pipeline(
            "knn", features=ALL_CANDIDATE_FEATURES,
            n_neighbors=5, weights="uniform", metric="euclidean",
        ),
    }
    baseline_results = {}
    for bname, bpipe in baselines_12.items():
        baseline_results[bname] = _eval_model_metrics(
            bpipe, X_train_all, y_train, X_test_all, y_test, g_train, kf, gkf,
        )

    # 5-fold GridSearchCV parameter grids on reduced 9 features
    param_spaces = {
        "svr": {
            "estimator": build_market_pipeline("svr", features=MARKET_FEATURES),
            "params": [
                {"regressor__regressor__kernel": ["linear"],
                 "regressor__regressor__C": [0.1, 1.0, 10.0]},
                {"regressor__regressor__kernel": ["rbf"],
                 "regressor__regressor__C": [0.1, 1.0, 10.0],
                 "regressor__regressor__gamma": ["scale", "auto"]},
            ],
            "prefix": "regressor__regressor__",
        },
        "random_forest": {
            "estimator": build_market_pipeline("random_forest", features=MARKET_FEATURES),
            "params": {
                "regressor__n_estimators": [10, 100, 500],
                "regressor__max_depth": [None, 3, 5, 10],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__max_features": ["sqrt", 1.0],
            },
            "prefix": "regressor__",
        },
        "knn": {
            "estimator": build_market_pipeline("knn", features=MARKET_FEATURES),
            "params": {
                "regressor__n_neighbors": [3, 5, 7, 9],
                "regressor__weights": ["uniform", "distance"],
                "regressor__metric": ["euclidean", "manhattan"],
            },
            "prefix": "regressor__",
        },
    }

    results = {}
    fitted_pipes = {}
    test_preds = {}
    grid_trials = {}

    if verbose:
        print(f"\n5-Fold GridSearchCV on {len(MARKET_FEATURES)} reduced features ({len(X_train):,} train / {len(X_test):,} untouched test)...")

    for kind in models:
        if kind not in param_spaces:
            raise ValueError(f"unknown model '{kind}' for GridSearchCV tuning")
        cfg = param_spaces[kind]
        t0 = time.time()
        search = GridSearchCV(
            cfg["estimator"],
            cfg["params"],
            cv=kf,
            scoring={"r2": "r2", "neg_mse": "neg_mean_squared_error"},
            refit="r2",
            return_train_score=True,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        pred = best_pipe.predict(X_test)
        dt = time.time() - t0

        # GroupKFold generalization check on X_train using the winning configuration
        gkf_scores = cross_validate(
            best_pipe, X_train, y_train, cv=gkf, groups=g_train, scoring="r2", n_jobs=-1,
        )
        group_cv_r2 = float(gkf_scores["test_score"].mean())

        cv_df = pd.DataFrame(search.cv_results_)
        best_idx = int(search.best_index_)
        train_r2 = float(cv_df.loc[best_idx, "mean_train_r2"])
        cv_r2 = float(search.best_score_)
        cv_r2_std = float(cv_df.loc[best_idx, "std_test_r2"])
        cv_mse_lakh2 = float(-cv_df.loc[best_idx, "mean_test_neg_mse"] / (LAKH ** 2))

        test_r2 = float(r2_score(y_test, pred))
        test_mse_lakh2 = float(mean_squared_error(y_test, pred) / (LAKH ** 2))
        test_mae = float(mean_absolute_error(y_test, pred))
        test_rmse = float(root_mean_squared_error(y_test, pred))
        test_mape = float(mean_absolute_percentage_error(y_test, pred))

        prefix = cfg["prefix"]
        best_params = {k.replace(prefix, ""): v for k, v in search.best_params_.items()}

        trial_rows = []
        for _, row in cv_df.sort_values("mean_test_r2", ascending=False).iterrows():
            clean_p = {k.replace(prefix, ""): v for k, v in row["params"].items()}
            trial_rows.append({
                "params": clean_p,
                "train_r2": float(row["mean_train_r2"]),
                "cv_r2": float(row["mean_test_r2"]),
                "cv_r2_std": float(row["std_test_r2"]),
                "cv_mse_lakh2": float(-row["mean_test_neg_mse"] / (LAKH ** 2)),
                "gap": float(row["mean_train_r2"] - row["mean_test_r2"]),
                "rank": int(row["rank_test_r2"]),
            })
        grid_trials[kind] = trial_rows

        fitted_pipes[kind] = best_pipe
        test_preds[kind] = pred
        results[kind] = {
            "best_params": best_params,
            "train_r2": train_r2,
            "cv_r2": cv_r2,
            "cv_r2_std": cv_r2_std,
            "cv_mse_lakh2": cv_mse_lakh2,
            "group_cv_r2": group_cv_r2,
            "gap": float(train_r2 - cv_r2),
            "test_r2": test_r2,
            "test_mse_lakh2": test_mse_lakh2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "fit_time_sec": round(dt, 1),
        }

        if verbose:
            print(
                f"  {kind:<15} Train R2: {train_r2:.4f} | 5-Fold CV R2: {cv_r2:.4f} | "
                f"CV MSE: {cv_mse_lakh2:.3f} L2 | GroupCV R2: {group_cv_r2:.4f} | "
                f"Test R2: {test_r2:.4f} | Test MAE: Rs {test_mae:,.0f} ({dt:.1f}s)"
            )
            print(f"    best_params: {best_params}")

    winner = max(results, key=lambda k: results[k]["cv_r2"])
    best = fitted_pipes[winner]
    best_pred = test_preds[winner]

    comparison_rows = []
    for name, m in results.items():
        comparison_rows.append({
            "model": name,
            "train_r2": m["train_r2"],
            "cv_r2": m["cv_r2"],
            "cv_mse_lakh2": m["cv_mse_lakh2"],
            "group_cv_r2": m["group_cv_r2"],
            "test_r2": m["test_r2"],
            "test_mse_lakh2": m["test_mse_lakh2"],
            "test_mae": m["test_mae"],
            "test_rmse": m["test_rmse"],
            "test_mape": m["test_mape"],
            "fit_time_sec": m["fit_time_sec"],
            "best_params": m["best_params"],
        })
    comparison_df = pd.DataFrame(comparison_rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)

    meta = {
        "market_model": winner,
        "cv": f"5-Fold GridSearchCV (KFold n_splits={cv_splits}, random_state={RANDOM_STATE}) + GroupKFold verification",
        "tuned": True,
        "features": list(MARKET_FEATURES),
        "features_count": len(MARKET_FEATURES),
        "dropped_features": list(DROPPED_LOW_CORR_FEATURES),
        "results": results,
        "test_metrics": results[winner],
        "comparison": comparison_df.to_dict(orient="records"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rows": int(len(features)),
        "year_min": int(features["year"].min()) if "year" in features else None,
        "year_max": int(features["year"].max()) if "year" in features else None,
        "brands": sorted(features["brand"].unique().tolist()),
        "models_known": sorted(features.loc[features["model"] != "other", "model"].unique().tolist()),
        "optimization": {
            "target_correlations": {k: float(v) for k, v in ranked_corr.items()},
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "feature_reduction": reduction_df.to_dict(orient="records") if not reduction_df.empty else [],
            "baselines_12_features": baseline_results,
            "grid_trials": grid_trials,
        },
    }

    if verbose:
        print(f"\nWinner: {winner} (5-Fold CV R2 = {results[winner]['cv_r2']:.4f}, Untouched Test R2 = {results[winner]['test_r2']:.4f})")

    return best, meta, (X_test, y_test, best_pred), comparison_df


def train_market(features: pd.DataFrame, *, test_size: float = 0.2,
                 cv_splits: int = 5, tune: bool = False,
                 models: tuple[str, ...] = ("svr", "random_forest", "knn"),
                 verbose: bool = True):
    """
    Fit the market model on the reduced 9-feature subset and report leak-free metrics.
    """
    if tune:
        best, meta, preds, _ = tune_market(
            features, test_size=test_size, cv_splits=cv_splits,
            models=models, verbose=verbose,
        )
        return best, meta, preds

    X = features[MARKET_FEATURES]
    y = features["price"].astype(float)
    groups = features["model"]

    X_train, X_test, y_train, y_test, g_train, _ = train_test_split(
        X, y, groups, test_size=test_size, random_state=RANDOM_STATE, shuffle=True,
    )

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    gkf = GroupKFold(n_splits=cv_splits)

    results = {}
    for kind in models:
        pipe = build_market_pipeline(kind, features=MARKET_FEATURES)
        results[kind] = _eval_model_metrics(pipe, X_train, y_train, X_test, y_test, g_train, kf, gkf)
        if verbose:
            r = results[kind]
            print(
                f"{kind:<15} CV R2 {r['cv_r2']:.4f}   GroupCV {r['group_cv_r2']:.4f}   "
                f"test R2 {r['test_r2']:.4f}   MAE Rs {r['test_mae']:,.0f}   MAPE {r['test_mape']:.1%}"
            )

    winner = max(results, key=lambda k: results[k]["cv_r2"])
    best = build_market_pipeline(winner, features=MARKET_FEATURES).fit(X_train, y_train)
    pred = best.predict(X_test)

    meta = {
        "market_model": winner,
        "cv": f"5-Fold CV (n_splits={cv_splits}) + GroupKFold verification",
        "tuned": False,
        "features": list(MARKET_FEATURES),
        "features_count": len(MARKET_FEATURES),
        "dropped_features": list(DROPPED_LOW_CORR_FEATURES),
        "results": results,
        "test_metrics": results[winner],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rows": int(len(features)),
        "year_min": int(features["year"].min()) if "year" in features else None,
        "year_max": int(features["year"].max()) if "year" in features else None,
        "brands": sorted(features["brand"].unique().tolist()),
        "models_known": sorted(features.loc[features["model"] != "other", "model"].unique().tolist()),
    }
    if verbose:
        print(f"\nwinner: {winner}")
    return best, meta, (X_test, y_test, pred)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_market(pipeline, meta: dict, path: str | Path,
                comparison_df: pd.DataFrame | None = None) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

    opt = meta.get("optimization")
    meta_slim = {k: v for k, v in meta.items() if k != "optimization"}
    path.with_suffix(".json").write_text(json.dumps(meta_slim, indent=2))

    if opt is not None:
        opt_path = path.parent / "optimization_results.json"
        opt_path.write_text(json.dumps(opt, indent=2))

    if comparison_df is not None:
        comp_csv = path.parent / "model_comparison.csv"
        comp_json = path.parent / "model_comparison.json"
        comparison_df.to_csv(comp_csv, index=False)
        comp_json.write_text(json.dumps(meta_slim.get("comparison", []), indent=2))
    return path


def load_market(path: str | Path):
    import joblib
    path = Path(path)
    model = joblib.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return model, meta

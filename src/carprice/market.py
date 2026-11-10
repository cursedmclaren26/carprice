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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate, train_test_split
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
def build_market_pipeline(model: str = "hist_gb") -> Pipeline:
    """
    Preprocessor + estimator.

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
        est = Ridge(alpha=3.0)
    elif model == "hist_gb":
        est = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"unknown market model '{model}'")
    return Pipeline([("preprocessor", pre), ("regressor", est)])


def train_market(features: pd.DataFrame, *, test_size: float = 0.2,
                 cv_splits: int = 5, verbose: bool = True):
    """
    Fit the market model and report honest metrics.

    Cross-validation is **grouped by model name**, so a fold never validates on a car
    whose near-identical siblings it just trained on. That is the leak that makes
    listing-based price models look better than they are on a genuinely new car.
    """
    X = features[MARKET_FEATURES]
    y = features["price"].astype(float)
    groups = features["model"]

    X_train, X_test, y_train, y_test, g_train, _ = train_test_split(
        X, y, groups, test_size=test_size, random_state=RANDOM_STATE, shuffle=True)

    results = {}
    for kind in ("ridge", "hist_gb"):
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
            "test_mape": float(mean_absolute_percentage_error(y_test, pred)),
        }
        if verbose:
            r = results[kind]
            print(f"{kind:<10} grouped-CV R2 {r['cv_r2']:.3f} +/- {r['cv_r2_std']:.3f}"
                  f"   test R2 {r['test_r2']:.3f}   MAE Rs {r['test_mae']:,.0f}"
                  f"   MAPE {r['test_mape']:.1%}")

    winner = max(results, key=lambda k: results[k]["cv_r2"])
    best = build_market_pipeline(winner).fit(X_train, y_train)
    pred = best.predict(X_test)

    meta = {
        "market_model": winner,
        "cv": f"GroupKFold({cv_splits}) grouped by model name",
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
def save_market(pipeline, meta: dict, path: str | Path) -> Path:
    import joblib, json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load_market(path: str | Path):
    import joblib, json
    path = Path(path)
    model = joblib.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return model, meta

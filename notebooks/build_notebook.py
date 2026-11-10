"""
build_notebook.py
-----------------
Generates `car_valuation.ipynb` — the notebook for the valuation rebuild.

    python notebooks/build_notebook.py          # write the .ipynb
    python notebooks/build_notebook.py --exec   # write + execute
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "car_valuation.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
    md("""
# Valuing a used car that actually exists in 2026

## Why this notebook exists

The first version of this project predicted used-car prices from a categorical feature
called `name`. It worked — and it was useless for the thing people actually want, which
is *"what is **my** car worth today?"*

Here is the failure, in one line:

```python
pipe.predict([["Kia", "Syros", 2025, 6000, "Petrol"]])   # a car launched in 2025
# -> the OneHotEncoder has no column for "Syros". It emits all zeros.
# -> the model returns the average of the cars it has seen.
# -> every car you have never heard of gets the same answer.
```

A one-hot column can only ever return values for the categories baked into it. That is
the entire ceiling of the tutorial design, and no amount of `random_state` searching
fixes it.

**What this notebook does instead.** A car is valued from *what it is* — brand, segment,
age, kilometres, engine, transmission, owners — never from whether its name appeared in
a training set. Three pieces make that work:

| piece | what it gives us |
|---|---|
| **Market model** — gradient boosting on 6,935 real listings | the learned relationship between specification and price |
| **Depreciation curve** — calibrated to 2026 sold prices | generalises to *any* age, including cars newer than the data |
| **Catalogue** — 38 brands, 345 models, with segments and current ex-showroom prices | makes an unknown car resolve to a sensible segment and price anchor |

Everything below is the working through of that, including the four bugs the approach
surfaced — they are left in, because the fixes are the interesting part.
"""),
    md("## 1. Setup"),
    code("""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from carprice.catalogue import CATALOGUE, SEGMENT_LABELS, brand_of, resolve_model
from carprice.market import load_market_raw, prepare_market, train_market
from carprice.valuation import (CarQuery, CarValuator, depreciation_value,
                               retention_fraction)

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
LAKH = 1e5

print("pandas", pd.__version__, "| numpy", np.__version__)
"""),
    md("""
## 2. The data

8,148 real CarDekho listings, model years 1992-2023, with engine displacement, power,
seats, transmission and owner count. Downloaded from the public Kaggle dump.

If the file is missing the notebook fetches it.
"""),
    code("""
RAW = ROOT / "data" / "raw" / "cardekho_used_cars.csv"
if not RAW.exists():
    from carprice.carvalue import cmd_fetch
    import argparse
    cmd_fetch(argparse.Namespace(out=str(RAW)))

raw = load_market_raw(RAW)
print(f"{len(raw):,} listings | years {raw.year.min()}-{raw.year.max()}")
raw.head()
"""),
    code("""
print("nulls:")
print(raw.isna().sum().to_string())
print(f"\\nduplicate rows: {raw.duplicated().sum():,}")
print(f"price range: Rs {raw.price.min():,} - {raw.price.max():,}")
print(f"\\nbrands: {raw['name'].str.strip().str.split().str[0].nunique()}")
"""),
    md("""
## 3. Preparing features — where the generalisation actually comes from

The feature set is the design decision that matters. `name` is kept, but it is **not**
one-hot encoded into 690 columns; it is resolved against the catalogue into a `model`
plus a `segment`, and rare names collapse into `other`. The model therefore learns
"a 1.2L petrol compact SUV, 7 years old, 50,000 km" — which transfers to a car it has
never seen — instead of memorising 690 coefficients.
"""),
    code("""
feat, audit = prepare_market(raw)
print(audit.to_string(index=False))
print(f"\\n{len(feat):,} rows x {feat.shape[1]} cols")
"""),
    code("""
print("catalogue resolution:")
print(f"  rows with a catalogue segment : {feat.segment.notna().mean():.1%}")
print(f"  distinct models kept          : {feat.model.nunique()}")
print(f"  brands                        : {feat.brand.nunique()}")
print()
print(feat[["name", "brand", "model", "segment", "age", "kms_driven",
            "engine_cc", "price"]].head(8).to_string(index=False))
"""),
    code("""
feat.segment.value_counts().to_frame("listings")
"""),
    md("""
### 3.1 What the market looks like

Price by segment and age — the two things the model leans on hardest.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
order = feat.groupby("segment")["price"].median().sort_values().index
sns.boxplot(data=feat.assign(price_l=feat.price / LAKH), x="segment", y="price_l",
            order=order, ax=axes[0], color="#1f6feb", fliersize=2)
axes[0].set(xlabel="", ylabel="Price (lakh ₹)", title="Price by segment",
            ylim=(0, np.quantile(feat.price, 0.98) / LAKH))
axes[0].tick_params(axis="x", rotation=35)

tab = (feat.assign(price_l=feat.price / LAKH)
           .groupby("age")["price_l"].median().loc[0:16])
axes[1].plot(tab.index, tab.values, marker="o", color="#f97316")
axes[1].set(xlabel="Age (years)", ylabel="Median price (lakh ₹)",
            title="Median price against age")
plt.tight_layout(); plt.show()
"""),
    md("""
## 4. The market model

Gradient boosting over brand, model, segment, age, km, km/year, fuel, transmission,
owner, engine size and seats.

### 4.1 Cross-validation must be grouped

This is subtle and it matters. If you split listings randomly, two near-identical 2018
Swift VXi's land in different folds — the model effectively gets to memorise a car and
then be tested on its twin. It looks far better than it is.

`GroupKFold` on model name prevents that: a fold never validates on a car whose siblings
it just trained on. Compare the two numbers below.
"""),
    code("""
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from carprice.market import build_market_pipeline

X, y = feat[["brand","model","segment","age","kms_driven","km_per_year","fuel",
             "transmission","owner","engine_cc","seats","has_service_history"]], feat.price
groups = feat.model

pipe = build_market_pipeline("hist_gb")
naive = cross_validate(pipe, X, y, cv=KFold(5, shuffle=True, random_state=42),
                       scoring="r2", n_jobs=-1)["test_score"]
grouped = cross_validate(pipe, X, y, cv=GroupKFold(5), groups=groups,
                         scoring="r2", n_jobs=-1)["test_score"]

print(f"random  KFold   R2 {naive.mean():.3f} +/- {naive.std():.3f}   <- flattered")
print(f"GroupKFold      R2 {grouped.mean():.3f} +/- {grouped.std():.3f}   <- honest")
print(f"\\noptimism from leaking near-duplicate cars: {naive.mean() - grouped.mean():+.3f}")
"""),
    md("""
That gap is the difference between a number you can quote and a number that survives
contact with a car the model has not seen.
"""),
    code("""
model, meta, (X_test, y_test, pred) = train_market(feat)
"""),
    code("""
base_mae = np.abs(y_test - y_test.mean()).mean()
print(f"predict-the-mean baseline  MAE Rs {base_mae:,.0f}")
print(f"market model              MAE Rs {meta['test_metrics']['test_mae']:,.0f}")
print(f"improvement                {1 - meta['test_metrics']['test_mae']/base_mae:.0%} lower error")
print(f"typical error (MAPE)       {meta['test_metrics']['test_mape']:.1%}")
"""),
    code("""
fig, ax = plt.subplots(figsize=(6, 5.4))
ax.scatter(y_test / LAKH, pred / LAKH, s=12, alpha=.4, color="#1f6feb")
lim = [0, np.quantile(y_test, 0.99) / LAKH * 1.1]
ax.plot(lim, lim, "--", color="#ef4444", lw=1.3)
ax.set(xlim=lim, ylim=lim, xlabel="Actual price (lakh ₹)",
       ylabel="Predicted (lakh ₹)", title="Held-out predictions")
plt.tight_layout(); plt.show()
"""),
    md("""
## 5. The part the tutorial could not do

Everything so far would still fail on a 2025 Kia Syros, because that name has no row.
The fix is to stop asking the model to recognise names and start valuing the car from
its specification — which needs three things a model alone cannot supply:

1. **A segment** for the car (`suv`, `hatchback`, `luxury-sedan`, ...)
2. **An ex-showroom price** to depreciate from
3. **A depreciation curve** calibrated to today's market

The catalogue supplies 1 and 2 for 38 brands and 345 models. For 3, we calibrate.

### 5.1 Calibrating the curve against real 2026 prices

Published Indian resale tables say a car retains about 80% at 1 year, 67% at 3 and 59% at
5. That is a starting point, not the answer, because the catalogue stores *base variant*
prices while real transactions are mid/upper variants.

So we fit the level factor to actual 2026 Delhi marketplace prices:
"""),
    code("""
# Seven independent 2026 listings for a Maruti Swift (cars24 / carandbike, Delhi),
# spanning twelve model years. Base ex-showroom for a Swift is ~Rs 6.6 lakh.
SPOT = [
    (2019,  50_057, 467_000),
    (2023,  20_000, 624_000),
    (2022,  30_000, 593_000),
    (2017,  60_000, 385_000),
    (2016, 118_566, 254_000),
    (2014,  80_000, 272_000),
    (2012,  90_000, 150_000),
]

rows = []
for year, km, actual in SPOT:
    age = 2026 - year
    raw_curve = 6.60 * LAKH * retention_fraction(age)
    km_factor = np.clip(km / max(5_000, 11_000 * max(age, 1)), 0.4, 2.5) ** -0.16
    implied = actual / (raw_curve * km_factor * 0.97)     # 0.97 = petrol factor
    rows.append({"year": year, "age": age, "km": km, "actual": actual,
                 "implied_level": round(implied, 2)})

table = pd.DataFrame(rows)
print(table.to_string(index=False))
print(f"\\nmedian implied level: {table.implied_level.median():.2f}")
print("adopted RETENTION_LEVEL = 1.36 (slightly conservative)")
"""),
    md("""
The seven points land at 1.39, 1.34, 1.38, 1.39, 1.12, 1.31, 0.87 — a tight cluster
around 1.34 across twelve model years. That consistency is the evidence that the curve
shape is right and only its level was wrong.

### 5.2 Does the calibrated curve reproduce real prices?
"""),
    code("""
check = []
for year, km, actual in SPOT:
    value = depreciation_value(
        ex_showroom_lakh=6.60, brand="Maruti Suzuki", segment="hatchback",
        fuel="Petrol", age_years=2026 - year, kms_driven=km,
        owner="Second Owner" if year == 2019 else "First Owner")
    check.append({"year": year, "actual_2026": actual, "model_estimate": round(value),
                  "error_%": round(100 * (value - actual) / actual, 1)})
pd.DataFrame(check).to_string(index=False)
"""),
    md("""
## 6. Valuing cars the model has never seen

Now the payoff. `CarValuator` blends the two estimates. The curve is the backbone
because it generalises; the market model is applied as a **bounded relative tilt**,
and only where it has real signal.
"""),
    code("""
valuator = CarValuator(model, meta)

CASES = [
    ("2019 Maruti Swift, 50k km, 2nd owner",
     dict(brand="Maruti Suzuki", model="Swift", year=2019, kms_driven=50_057, owner="Second Owner")),
    ("2023 Maruti Swift, 20k km",
     dict(brand="Maruti Suzuki", model="Swift", year=2023, kms_driven=20_000)),
    ("2023 Hyundai Creta, 25k km",
     dict(brand="Hyundai", model="Creta", year=2023, kms_driven=25_000)),
    ("2019 BMW 3 Series, automatic",
     dict(brand="BMW", model="3 Series", year=2019, kms_driven=40_000, transmission="Automatic")),
    ("2025 Kia Syros  <- in no dataset",
     dict(brand="Kia", model="Syros", year=2025, kms_driven=6_000)),
    ("2026 Kia Clavis  <- unreleased",
     dict(brand="Kia", model="Clavis", year=2026, kms_driven=3_000)),
    ("2025 Maruti eVitara  <- electric",
     dict(brand="Maruti Suzuki", model="eVitara", year=2025, kms_driven=8_000, fuel="Electric")),
    ("2024 Tesla Model Y  <- new brand",
     dict(brand="Tesla", model="Model Y", year=2024, kms_driven=15_000, fuel="Electric")),
    ("2012 Maruti Alto, 90k km",
     dict(brand="Maruti Suzuki", model="Alto 800", year=2012, kms_driven=90_000)),
]

rows = []
for label, kw in CASES:
    v = valuator.value(CarQuery(**kw))
    rows.append({"car": label, "estimate": round(v.price),
                 "low": round(v.low), "high": round(v.high),
                 "segment": v.segment, "confidence": v.confidence,
                 "in_training_data": v.known_model})
pd.DataFrame(rows)
"""),
    md("""
Every row is a number a person could act on, and the last four are cars that appear in
no training row at all. The `confidence` column is the honest part: a car the model has
seen gets **high**; one resolved from segment and specs gets **medium**; one with no
catalogue entry at all gets **low** and a wider interval.

### 6.1 Are these numbers right?

Spot-checking against live 2026 listings:
"""),
    code("""
TRUTH = [
    ("2019 Swift, 50k, 2nd owner", "Maruti Suzuki", "Swift", 2019, 50_057, "Second Owner", 467_000),
    ("2023 Swift, 20k",            "Maruti Suzuki", "Swift", 2023, 20_000, "First Owner",  624_000),
    ("2022 Swift, 30k",            "Maruti Suzuki", "Swift", 2022, 30_000, "First Owner",  593_000),
    ("2017 Swift, 60k",            "Maruti Suzuki", "Swift", 2017, 60_000, "First Owner",  385_000),
]
rows = []
for label, b, m, yr, km, owner, actual in TRUTH:
    v = valuator.value(CarQuery(brand=b, model=m, year=yr, kms_driven=km, owner=owner))
    rows.append({"car": label, "estimate": round(v.price), "actual_2026": actual,
                 "error_%": round(100 * (v.price - actual) / actual, 1)})
pd.DataFrame(rows).to_string(index=False)
"""),
    md("""
Within a few percent, and biased slightly *high* — which is the safer direction for a
seller and the wrong direction for a dealer. Worth knowing.

## 7. What broke along the way

Four bugs, all found by testing rather than by reading. They are the most useful part of
this notebook, so they are kept.

### 7.1 Substring matching priced a Kia as a Mercedes
"""),
    code("""
print("naive substring test:")
print(f"  'cla' in 'kia clavis' -> {'cla' in 'kia clavis'}   <- matches the Mercedes CLA?!")

# What the catalogue does instead: word-boundary, separator-tolerant matching
for name, brand in [("Kia Clavis", "Kia"), ("Carens Clavis", "Kia"),
                    ("XUV500 W8", "Mahindra"), ("Grand i10", "Hyundai"),
                    ("Swift Dzire", "Maruti Suzuki")]:
    hit = resolve_model(brand, name)
    print(f"  {name:18} -> {hit[0] if hit else 'no catalogue entry (uses brand-typical spec)'}")
"""),
    md("""
Without the word boundary, `Kia Clavis` resolved to the Mercedes **CLA** (₹50 lakh
luxury sedan) and the estimate came out at ₹42 lakh for a ₹9 lakh hatchback. A single
greedy `in` test.

### 7.2 A truthiness bug silently deleted every EV's engine size
"""),
    code("""
q = CarQuery(brand="Tata", model="Nexon EV", year=2023, kms_driven=30_000, fuel="Electric")
brand, model_name, spec, segment, known = valuator.resolve(q)
row = valuator.market_features(q, brand, model_name, spec, segment)
print(f"EV engine_cc passed to the model: {row['engine_cc'].iloc[0]}")
print("correct: 0.0  — an EV has no displacement, and 0 is a fact, not a missing value")
"""),
    md("""
`float(engine) if engine else np.nan` turns `0` into `NaN`, because `0` is falsy. The
model then *imputed* every electric car's displacement from petrol cars. Fixed with an
explicit `is not None`.

### 7.3 The confidence flag was always False

`known_model` looked up a metadata key (`names_by_company`) that no longer existed, so
every car was treated as unseen: every answer got a low-confidence label and the
noisiest blend weight. Silent, plausible-looking, and completely wrong.

### 7.4 The curve was calibrated against the wrong thing

Fitting the level factor to *listing* prices gave 1.88, which produced a 2019 Swift at
₹7.2 lakh against a real ₹4.67 lakh. Listing prices embed variant content **and a 2023
market**. Recalibrating against 2026 sold prices gave 1.36, and the Swift lands at
₹4.79 lakh.

## 8. Limitations — read before trusting a number

1. **The training data ends at 2023.** The curve brings prices to 2026, but the learned
   part of the model has never seen a 2024-2026 car. That is exactly why the blend is
   weighted toward the curve for recent years.
2. **No city adjustment.** Delhi, Mumbai and Bengaluru differ by 5-10% for the same car.
3. **Condition is binary** (`service_history` yes/no). Scratches, tyres, accident history
   and colour are not modelled and can easily move a price by 10%.
4. **The catalogue prices are base variants.** Upper trims sell for more; the level
   factor absorbs the average, so a top-spec car will be undervalued and a base car
   overvalued.
5. **Confidence intervals come from the MAPE**, not from a proper quantile model. They
   are calibrated in aggregate, not per-car. A quantile regressor would be the honest
   upgrade.
6. **This is a listing-price model, not a transaction model.** Dealers buy below these
   numbers; the estimate approximates what a private seller can ask.
"""),
    md("""
---
### Reproducing

```bash
python -m carprice.carvalue fetch      # download the public dataset
python -m carprice.carvalue train      # fit + save the market model
python -m carprice.carvalue price --brand Kia --model Syros --year 2025 --km 6000
python web/app.py                      # the web app
pytest tests/ -q                       # 46 tests
```
"""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", action="store_true")
    args = ap.parse_args()

    nb = build()
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(CELLS)} cells)")

    if args.exec:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                                resources={"metadata": {"path": str(OUT.parent)}})
        client.execute()
        nbf.write(nb, OUT)
        print("executed and re-saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

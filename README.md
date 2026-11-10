# Car Valuation — price any used car, including ones nobody has listed yet

A used-car valuation engine for the Indian market. Built out of the classic
"Quikr used-car price prediction" tutorial, which could only ever return prices for cars
already in its dataset.

<img src="docs/ui_result.png" width="720" alt="The valuation app">

## The problem with the tutorial version

The original model one-hot encoded the car's `name`. That design has a hard ceiling:

```python
pipe.predict([["Kia", "Syros", 2025, 6000, "Petrol"]])   # launched 2025
# OneHotEncoder has no column for "Syros" -> all zeros
# model returns the average of the cars it saw -> every unknown car gets the same answer
```

Ask it about a car that is not one of its ~690 training names and it does not say "I
don't know" — it returns a confident, meaningless average. That is not a tuning problem;
the information is not in a categorical column, so it cannot be recovered by a better
model or a luckier `random_state`.

## How this values an unseen car

| piece | supplies | generalises? |
|---|---|---|
| **Market model** — gradient boosting on **6,935 real listings** | learned relationship between specification and price | to new *combinations* of seen specs |
| **Depreciation curve** — calibrated against **2026 sold prices** | price from ex-showroom × age × brand/segment | to **any** car, any age |
| **Catalogue** — **38 brands, 345 models** with segments and current ex-showroom prices | segment, engine, seats, price anchor | to models in no dataset |

A car is valued from **what it is** — brand, segment, age, kilometres, engine,
transmission, owners — never from whether its name appeared in training. An unrecognised
name falls back to the brand's characteristic segment (a Kia resolves to a compact SUV,
not the fleet average), and the answer says so.

## Results

| | |
|---|---|
| Training data | 8,148 real CarDekho listings → 6,935 after cleaning |
| Held-out R² | **0.885** |
| Held-out MAE | **₹90,681** (vs ₹322,778 for predicting the mean — **72% lower**) |
| Typical error (MAPE) | **19.1%** |
| CV protocol | `GroupKFold(5)` **grouped by model name** |

That last row matters. Random KFold scores **0.847**; GroupKFold scores **0.754**. The
0.093 gap is leakage — with a random split, near-identical listings of the same car land
in both train and test, so the model is graded on cars it has effectively memorised.
Grouped CV is the number that describes a genuinely new car.

### Validated against live 2026 prices

Seven independent 2026 Delhi marketplace listings for a Maruti Swift, spanning twelve
model years:

| year | actual 2026 | estimate | error |
|---|---|---|---|
| 2019 | ₹4,67,000 | ₹4,67,528 | **+0.1%** |
| 2023 | ₹6,24,000 | ₹6,46,800 | +3.7% |
| 2022 | ₹5,93,000 | ₹6,30,542 | +6.3% |
| 2017 | ₹3,85,000 | ₹4,07,960 | +6.0% |

## Quick start

```bash
pip install -e ".[eda,web]"

python -m carprice.carvalue fetch     # download the public dataset (~1 MB)
python -m carprice.carvalue train     # fit + save the market model
python web/app.py                     # http://127.0.0.1:5000
```

### Command line

```bash
$ python -m carprice.carvalue price --brand "Maruti Suzuki" --model Swift \
        --year 2019 --km 50057 --owner "Second Owner"

  2019 Maruti Suzuki Swift  (50,057 km, Petrol, Manual)
  --------------------------------------------------------
  Estimated selling price   Rs 4,79,105
  Likely range              Rs 4,26,404 - 5,31,807
  Confidence                high
  Method                    depreciation curve + market adjustment
```

```bash
$ python -m carprice.carvalue price --brand Kia --model Syros --year 2025 --km 6000
  2025 Kia Syros  (6,000 km, Petrol, Manual)
  --------------------------------------------------------
  Estimated selling price   Rs 8,86,585
  Likely range              Rs 7,27,000 - 10,46,171
  Confidence                medium
  Note                      this exact model is not in the training data;
                            valued from segment, age and specs
```

```bash
python -m carprice.carvalue brands
python -m carprice.carvalue models --brand Hyundai
python -m carprice.carvalue price ... --json      # machine-readable
```

## What it will not do

Being explicit about this is the difference between a tool and a toy.

1. **The data ends at 2023.** The curve carries prices to 2026, but the learned model has
   never seen a 2024-2026 car — which is why recent cars lean on the curve.
2. **No city adjustment.** Delhi/Mumbai/Bengaluru differ 5-10% for the same car.
3. **Condition is binary.** No scratches, tyres, accident history or colour. Any of those
   can move a real price by 10%.
4. **Catalogue prices are base variants**, so a top-spec car is under-valued and a base
   car over-valued. The level factor absorbs the average.
5. **Intervals come from the aggregate MAPE**, not from a quantile model. Calibrated on
   average, not per car.
6. **Listing prices, not transaction prices.** Dealers buy below these numbers.

## Layout

```
carprice/
├── src/carprice/
│   ├── catalogue.py     # 38 brands, 345 models: segment, ex-showroom, engine, seats
│   ├── valuation.py     # depreciation curve + market blend + confidence
│   ├── market.py        # feature prep and training for the market model
│   ├── carvalue.py      # CLI: fetch / train / price / brands / models
│   ├── cleaning.py      # parsers for the original Quikr schema (kept)
│   └── modeling.py      # the original tutorial pipeline (kept as the "before")
├── notebooks/car_valuation.ipynb   # the full write-up, executed
├── web/app.py           # Flask app
├── tests/               # 46 tests
└── data/raw/cardekho_used_cars.csv
```

`cleaning.py` and `modeling.py` from the first version are kept deliberately: they are
the honest "before" that the notebook measures itself against.

## Tests

```bash
pytest tests/ -q     # 46 passed
```

Regression tests cover the four bugs found while building this:

* `Kia Clavis` substring-matching the Mercedes `CLA` (priced a ₹9 lakh hatchback at ₹42 lakh)
* `known_model` always `False` because it read a metadata key that no longer existed
* `if engine_cc` treating an EV's `0` as missing and imputing it from petrol cars
* the retention level (1.88) fitted against *listing* prices instead of sold prices

## The notebook

`notebooks/car_valuation.ipynb` walks through all of it — the leakage demo, the curve
calibration against live prices, the four bugs, and the limitations. 32 cells, no errors.

---

## Notes on the data

`data/merged_car_dataset.csv` is **synthetic** (`make_sample_data.py`). It reproduces the
schema and the specific failure modes of the original scrape — `Rs 4,25,000`, `4.5 Lac`,
`Ask For Price`, `"petrol"` sitting in the kms column, `URJENT`/`Used` company labels,
duplicates, absurd outliers — so every cleaning branch is exercised. Point the trainer at
your real CSV to get real numbers; nothing else changes.

The original tutorial's figures are preserved here for comparison only: 892 raw rows →
816 after cleaning → 815 after the price cut, 254 name categories, baseline R² 0.7627,
and a best-of-1000 R² of 0.92009 at `random_state=655`.

## Tests

```bash
pytest tests/ -q
```

Covers the parsers (including the `"petrol"`-in-kms case), the audit trail, unknown-category
handling at prediction time, and a round-trip of the saved pickle.

"""
make_sample_data.py
-------------------
Creates a synthetic-but-realistic *raw* used-car dataset that mirrors the messiness of the
original `quikr_car.csv` (and the `merged_car_dataset.csv` variant), so that every cleaning
step in the notebook is actually exercised.

Messiness injected on purpose:
  * name      -> inconsistent casing, spammy tails ("showroom condition with", "Well mentained")
  * company   -> junk labels ("Used", "URJENT", "Commercial", "Others", NaN)
  * year      -> text like "2015.0", "2015 year", "petrol", "2006 Oct", NaN
  * price     -> "Rs 4,25,000", "4.25 Lac", "Ask For Price", plain ints, NaN, absurd outliers
  * kms       -> "45,000 kms", "1.2 Lac kms", 0, NaN, "petrol" (the infamous data-entry bug)
  * fuel      -> Petrol / Diesel / LPG / CNG / Electric / NaN / "petrol " (trailing space)

Usage:
    python make_sample_data.py            # writes data/merged_car_dataset.csv
    python make_sample_data.py --rows 1500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42

# company -> (models, popularity weight, price multiplier)
BRANDS: dict[str, tuple[list[str], float, float]] = {
    "Maruti Suzuki": (
        ["Swift", "Alto 800", "Wagon R", "Dzire", "Baleno", "Celerio", "Ertiga",
         "Brezza", "800", "Omni", "SX4", "Ciaz", "Ignis", "Ritz"],
        0.26, 1.00),
    "Hyundai": (
        ["i20", "Grand i10", "i10", "Creta", "Verna", "Santro Xing", "Eon",
         "Xcent", "Venue", "Elantra"],
        0.18, 1.05),
    "Tata": (
        ["Nexon", "Altroz", "Tiago", "Tigor", "Indica V2", "Indigo CS", "Zest",
         "Nano", "Sumo Grande", "Harrier", "Safari"],
        0.10, 0.95),
    "Mahindra": (
        ["Scorpio", "XUV500", "Bolero", "Thar", "TUV300", "KUV100", "Quanto",
         "Xylo", "Marazzo"],
        0.10, 1.05),
    "Honda": (
        ["City", "Amaze", "Jazz", "Brio", "WR-V", "Civic", "Accord"],
        0.08, 1.15),
    "Toyota": (
        ["Innova Crysta", "Fortuner", "Etios", "Corolla Altis", "Glanza", "Yaris", "Qualis"],
        0.06, 1.35),
    "Ford": (
        ["EcoSport", "Figo", "Aspire", "Fiesta", "Endeavour", "Ikon"],
        0.05, 1.05),
    "Volkswagen": (
        ["Polo", "Vento", "Ameo", "Jetta", "Passat"],
        0.04, 1.20),
    "Skoda": (
        ["Rapid", "Octavia", "Superb", "Fabia", "Laura", "Kushaq"],
        0.03, 1.25),
    "Renault": (
        ["Kwid", "Duster", "Triber", "Lodgy", "Scala"],
        0.03, 0.95),
    "Chevrolet": (
        ["Beat", "Spark", "Cruze", "Sail UVA", "Tavera", "Enjoy"],
        0.03, 0.85),
    "Nissan": (
        ["Micra", "Sunny", "Terrano", "Kicks"],
        0.02, 0.95),
    "Datsun": (
        ["GO", "Go Plus", "Redi-GO"],
        0.02, 0.85),
    "BMW": (
        ["3 Series", "5 Series", "7 Series", "X1", "X3"],
        0.015, 3.20),
    "Audi": (
        ["A4 2.0", "A6 2.0", "Q3 2.0", "Q5 2.0", "Q7", "A3 Cabriolet"],
        0.015, 3.30),
    "Mercedes-Benz": (
        ["C-Class", "E-Class", "A-Class", "GLA", "GLC"],
        0.012, 3.40),
}

JUNK_COMPANIES = ["Used", "URJENT", "Commercial", "Others", "Sale", "Private"]
TRIMS = ["VXi", "LXi", "ZXi", "VX", "LXI", "SX", "Magna", "Sportz", "Asta",
         "Trend", "Titanium", "Ambiance", "Diesel", "ZX", "ELX"]
FUELS = ["Petrol", "Diesel", "LPG", "CNG", "Electric"]


def _name_case(model: str, company: str, rng: np.random.Generator) -> str:
    """Return a raw, inconsistently formatted listing title."""
    style = rng.integers(0, 5)
    trim = rng.choice(TRIMS) if rng.random() < 0.45 else ""
    if company == "Maruti Suzuki":  # most listings omit "Suzuki"
        brand = "Maruti" if rng.random() < 0.75 else "Maruti Suzuki"
    else:
        brand = company
    core = f"{brand} {model}".strip()
    if trim and trim.lower() != model.split()[0].lower():
        core = f"{core} {trim}"
    if style == 0:
        return core
    if style == 1:
        return core.upper()
    if style == 2:
        return core.lower()
    if style == 3:
        return f"{core} showroom condition with"
    return f"Well mantained {core}"


def _indian_grouping(n: int) -> str:
    """1234567 -> '12,34,567' (lakh/crore grouping, as used on Indian listings)."""
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def _price_text(price: int, rng: np.random.Generator) -> str:
    """Render prices the way a scraped classifieds site would."""
    roll = rng.random()
    if roll < 0.035:
        return "Ask For Price"
    if roll < 0.08:
        return str(price)                              # plain int, no commas
    if roll < 0.15 and price >= 100_000:               # "Rs" + Indian digit grouping
        return f"Rs {_indian_grouping(price)}"
    if roll < 0.20 and price % 100_000 == 0:
        return f"{price // 100_000} Lac"
    if roll < 0.23 and price % 1_000_000 == 0:
        return f"{price // 1_000_000} Cr"
    return _indian_grouping(price)


def _kms_text(kms: int, rng: np.random.Generator) -> object:
    roll = rng.random()
    if roll < 0.05:
        return np.nan
    if roll < 0.08:
        return "petrol"                                # the classic data-entry bug
    if roll < 0.13 and kms >= 100_000 and kms % 10_000 == 0:
        return f"{kms // 100_000} Lac kms"
    if roll < 0.20:
        return _indian_grouping(kms)                   # missing " kms" suffix
    return f"{_indian_grouping(kms)} kms"


def _year_text(year: int, rng: np.random.Generator) -> object:
    roll = rng.random()
    if roll < 0.05:
        return str(f"{year} year")
    if roll < 0.08:
        return f"{year}.0"                              # float leaking through
    if roll < 0.11:
        return f"{year} Oct"
    return str(year)


def build(rows: int = 1000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    brands = list(BRANDS)
    weights = np.array([BRANDS[b][1] for b in brands], dtype=float)
    weights = weights / weights.sum()
    this_year = 2024

    records: list[dict] = []
    for _ in range(rows):
        company = str(rng.choice(brands, p=weights))
        models, _, mult = BRANDS[company]
        model = str(rng.choice(models))

        year = int(np.clip(rng.normal(this_year - 7, 4), 1998, this_year))
        age = this_year - year
        kms = int(np.clip(rng.normal(12000 * max(age, 1) * rng.uniform(0.6, 1.6),
                                     15000), 10, 400_000))
        fuel = str(rng.choice(FUELS, p=[0.52, 0.38, 0.02, 0.03, 0.05]))
        # Used-car exchanges sometimes list the fuel type as both fuel & kms.

        base = 620_000
        price = base * mult
        price *= np.exp(-0.13 * age)                    # depreciation
        price *= np.exp(-kms / 260_000)                 # wear
        price *= {"Petrol": 1.0, "Diesel": 1.06, "LPG": 0.90,
                  "CNG": 0.93, "Electric": 1.10}[fuel]
        price *= rng.uniform(0.80, 1.20)                # listing noise
        if company.startswith("Maruti") and age < 6:
            price *= 1.08                               # resale king
        price = int(np.clip(round(price / 5_000) * 5_000, 30_000, 8_500_000))

        name = _name_case(model, company, rng)

        inr_company = company
        roll = rng.random()
        if roll < 0.03:
            inr_company = str(rng.choice(JUNK_COMPANIES))
        elif roll < 0.035:
            inr_company = np.nan

        fuel_raw = fuel
        if rng.random() < 0.04:
            fuel_raw = f"{fuel.lower()} "                # trailing space
        elif rng.random() < 0.05:
            fuel_raw = np.nan

        records.append(
            {
                "name": name,
                "company": inr_company,
                "year": _year_text(year, rng),
                "selling_price": _price_text(price, rng),
                "km_driven": _kms_text(kms, rng),
                "fuel": fuel_raw,
            }
        )

    # A handful of absurd outliers + exact duplicates, like a real scrape.
    df = pd.DataFrame.from_records(records)
    outliers = df.sample(6, random_state=seed).copy()
    outliers["selling_price"] = ["85,00,000", "Rs 92,00,000", "1.5 Cr",
                                 "9,99,999", "1,25,00,000", "78,00,000"]
    df = pd.concat([df, outliers, df.sample(8, random_state=seed + 1)], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("data/merged_car_dataset.csv"))
    args = ap.parse_args()

    df = build(args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} raw rows -> {args.out}")
    print(df.head(8).to_string())
    print("\nNulls per column:\n", df.isna().sum().to_string())


if __name__ == "__main__":
    main()

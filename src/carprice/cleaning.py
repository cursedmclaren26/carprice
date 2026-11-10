"""
cleaning.py
-----------
One battle-tested cleaning routine for both source files.

The original project existed in two flavours whose schemas disagreed:

    quikr_car.csv            merged_car_dataset.csv
    ---------------------    ----------------------
    name                     name
    company                  company
    year                     year
    Price            <--->   selling_price
    kms_driven       <--->   km_driven
    fuel_type        <--->   fuel

`COLUMN_ALIASES` reconciles them so a single loader works for either file. The
cleaning steps are the ones from the tutorial (they were sound), but every step is
now explicit, counted, and logged into a `CleanReport` instead of silently dropping
rows with chained boolean masks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: canonical_name -> every spelling seen across the two source files
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name",),
    "company": ("company",),
    "year": ("year",),
    "price": ("Price", "price", "selling_price", "selling price"),
    "kms_driven": ("kms_driven", "km_driven", "kms driven", "km driven"),
    "fuel_type": ("fuel_type", "fuel", "fuel type"),
}

FEATURES = ["name", "company", "year", "kms_driven", "fuel_type"]
TARGET = "price"

#: A used hatchback above ~60 lakh is almost certainly a typo or a luxury/exotic
#: listing that would dominate the loss. The original notebook cut at 6,000,000;
#: that threshold is kept, but applied *after* parsing so it is comparable.
PRICE_CEILING = 6_000_000
YEAR_FLOOR = 1990

#: bucket that absorbs model names too rare to learn a coefficient from
OTHER_NAME = "other"

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_LAKH_RE = re.compile(r"(lac|lakh|lakhs|lacs|l\b)", re.I)
_CRORE_RE = re.compile(r"(cr|crore|crores)", re.I)


@dataclass
class CleanReport:
    """Row-by-row account of what cleaning removed and why."""

    raw_rows: int = 0
    steps: list[tuple[str, int, int]] = field(default_factory=list)  # (step, before, after)

    def log(self, step: str, before: int, after: int) -> None:
        self.steps.append((step, before, after))

    @property
    def final_rows(self) -> int:
        return self.steps[-1][2] if self.steps else self.raw_rows

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.steps, columns=["step", "rows_before", "rows_after"])
        df["rows_dropped"] = df["rows_before"] - df["rows_after"]
        return df

    def __str__(self) -> str:  # pragma: no cover - display helper
        lines = [f"raw rows: {self.raw_rows:,}"]
        for step, before, after in self.steps:
            lines.append(f"  {step:<42} {before:>5,} -> {after:>5,}  (-{before - after})")
        lines.append(f"clean rows: {self.final_rows:,}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> pd.DataFrame:
    """Read either source CSV and rename its columns to the canonical schema."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    lookup = {alias.strip().lower(): canon
              for canon, aliases in COLUMN_ALIASES.items() for alias in aliases}
    rename, unknown = {}, []
    for col in df.columns:
        canon = lookup.get(col.lower())
        if canon is None:
            unknown.append(col)
        else:
            rename[col] = canon
    df = df.rename(columns=rename)
    if unknown:
        df = df.drop(columns=unknown)

    missing = [c for c in (*FEATURES, TARGET) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {missing}. "
            f"Found: {list(pd.read_csv(path, nrows=0).columns)}"
        )
    return df


# --------------------------------------------------------------------------- #
# field parsers
# --------------------------------------------------------------------------- #
def parse_money(value: object) -> float:
    """'Rs 4,25,000' -> 425000.0 ; '4.5 Lac' -> 450000.0 ; 'Ask For Price' -> NaN."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    if not text or "ask" in text or "call" in text or "price on request" in text:
        return float("nan")

    match = _NUM_RE.search(text.replace(",", ""))
    if match is None:
        return float("nan")
    number = float(match.group(1))

    if _CRORE_RE.search(text):
        return number * 1_00_00_000
    if _LAKH_RE.search(text):
        return number * 1_00_000
    return number


def parse_kms(value: object) -> float:
    """'45,000 kms' -> 45000.0 ; '1.2 Lac kms' -> 120000.0 ; 'petrol' -> NaN."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    match = _NUM_RE.search(text.replace(",", ""))
    if match is None:
        return float("nan")
    number = float(match.group(1))
    return number * 1_00_000 if _LAKH_RE.search(text) else number


def parse_year(value: object) -> float:
    """'2015', '2015.0', '2015 year', '2006 Oct' -> 2015.0, 2015.0, 2015.0, 2006.0."""
    if pd.isna(value):
        return float("nan")
    text = str(value)
    match = _YEAR_RE.search(text)
    if match is None:
        return float("nan")
    return float(match.group(1))


_SPAM_WORDS = r"\b(showroom|condition|with|well|mantained|maintained|urgent|sale|for)\b"


def _clean_words(value: object) -> list[str]:
    """Lower-case tokens with listing spam removed."""
    if pd.isna(value):
        return []
    text = re.sub(_SPAM_WORDS, " ", str(value), flags=re.I)
    return re.sub(r"\s+", " ", text).strip().split()


def infer_trim_vocabulary(names: pd.Series, company_words: set[str],
                          min_models: int = 5) -> set[str]:
    """
    Discover variant/trim tokens (VXi, ZXi, Sportz, Magna, ...) from the data.

    Insight: a *model* name appears under one brand and one model head, whereas a
    *trim* appears across many different models ("Swift VXi", "Baleno VXi",
    "Dzire VXi"). So any token that shows up alongside `min_models` distinct
    (brand, model-head) pairs is a trim, not a model.

    This is what lets 'Maruti Baleno VXi' and 'Maruti Baleno ZXi' collapse into a
    single category, which is what the model actually needs to learn from.
    """
    seen: dict[str, set[tuple[str, str]]] = {}
    for words in names.map(_clean_words):
        head = next((w for w in words if w.lower() not in company_words), None)
        if head is None:
            continue
        for tok in words:
            low = tok.lower()
            if low == head.lower() or low in company_words or low.isdigit():
                continue
            seen.setdefault(low, set()).add((words[0].lower(), head.lower()))
    return {tok for tok, models in seen.items() if len(models) >= min_models}


def _title_token(word: str) -> str:
    if word.upper() in _ACRONYMS:
        return _ACRONYMS[word.upper()]
    return word.title() if (word.islower() or word.isupper()) else word


def canonical_name(value: object, trims: set[str] | None = None,
                   company_words: set[str] | None = None) -> str:
    """
    Collapse a raw listing title to a learnable *model* name.

    Drops listing spam and trim tokens, drops the brand tokens (the brand is
    already its own feature, so keeping it here would double-count), and keeps at
    most three model tokens.

    The tutorial used `" ".join(x.split()[:3])`, which keeps the noise
    ("Maruti Suzuki showroom") and treats every trim as a separate category.
    """
    words = _clean_words(value)
    if trims:
        words = [w for w in words if w.lower() not in trims]
    if company_words:
        words = [w for w in words if w.lower() not in company_words]
    words = words[:3]
    return " ".join(_title_token(w) for w in words)


#: brands whose spelling is not title-case, so `str.title()` must not touch them
_ACRONYMS = {"BMW": "BMW", "TVS": "TVS", "JEEP": "Jeep", "MINI": "MINI",
             "MG": "MG", "KIA": "Kia", "BYD": "BYD", "DC": "DC"}


def normalise_company(value: object) -> str:
    """Fix casing without destroying acronyms ('BMW' must not become 'Bmw')."""
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return ""
    if text.upper() in _ACRONYMS:
        return _ACRONYMS[text.upper()]
    # 'MARUTI SUZUKI' / 'maruti suzuki' -> 'Maruti Suzuki', but leave mixed case alone
    return text.title() if (text.isupper() or text.islower()) else text


def normalise_fuel(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().title().replace("Cng", "CNG").replace("Lpg", "LPG")


# --------------------------------------------------------------------------- #
# main cleaning routine
# --------------------------------------------------------------------------- #
def clean_cars(df: pd.DataFrame, *, report: CleanReport | None = None,
               min_year: int = YEAR_FLOOR,
               max_price: float = PRICE_CEILING,
               min_name_freq: int = 2) -> tuple[pd.DataFrame, CleanReport]:
    """
    Clean a raw frame in the canonical schema and return (clean_df, report).

    Steps (each one logged with row counts):
      1. coerce year / price / kms from text
      2. drop rows with no usable year, price or kms
      3. keep only plausible model years
      4. normalise name, company, fuel_type strings
      5. drop junk company labels
      6. pool rare model names into "other" (see `pool_rare`)
      7. collapse duplicate listings
      8. remove absurd prices (>= max_price), e.g. typos and exotics
      9. dtype fix-ups: year -> int, price -> int
    """
    rep = report or CleanReport(raw_rows=len(df))
    rep.raw_rows = len(df)
    df = df.copy()
    rep.log("raw input", len(df), len(df))

    for col in ("year", "price", "kms_driven"):
        parser = {"year": parse_year, "price": parse_money, "kms_driven": parse_kms}[col]
        df[col] = df[col].map(parser)

    before = len(df)
    df = df.dropna(subset=["year", "price", "kms_driven"])
    rep.log("parsed year/price/kms (dropped unparseable)", before, len(df))

    before = len(df)
    df = df[df["year"].between(min_year, pd.Timestamp.now().year)]
    rep.log(f"kept model year in [{min_year}, now]", before, len(df))

    df["company"] = df["company"].map(normalise_company)
    df["fuel_type"] = df["fuel_type"].map(normalise_fuel)

    before = len(df)
    df = df[~df["company"].str.lower().isin(
        ["used", "urjent", "commercial", "others", "sale", "private", "nan", ""])]
    rep.log("dropped junk company labels", before, len(df))

    # Learn the trim vocabulary from the surviving, well-formed rows, then collapse names.
    company_words = {w.lower() for c in df["company"].unique() for w in c.split()}
    trims = infer_trim_vocabulary(df["name"], company_words)
    df["name"] = df["name"].map(lambda n: canonical_name(n, trims, company_words))

    before = len(df)
    df = df[df["name"] != ""]
    rep.log("dropped rows with no usable name", before, len(df))

    before = len(df)
    counts = df["name"].value_counts()
    rare = counts[counts < min_name_freq].index
    df.loc[df["name"].isin(rare), "name"] = OTHER_NAME
    rep.log(f"pooled names appearing < {min_name_freq}x into '{OTHER_NAME}'",
            before, len(df))

    before = len(df)
    df = df.drop_duplicates(subset=["name", "company", "year", "kms_driven",
                                    "fuel_type", "price"])
    rep.log("dropped exact duplicate listings", before, len(df))

    before = len(df)
    df = df[df["price"] < max_price]
    rep.log(f"dropped price >= {max_price:,.0f}", before, len(df))

    df = df.astype({"year": int, "price": int, "kms_driven": int})
    df = df.reset_index(drop=True)
    df = df[["name", "company", "year", "price", "kms_driven", "fuel_type"]]
    return df, rep

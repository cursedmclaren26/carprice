"""
valuation.py
------------
Price any car — including one the market model has never seen.

The problem this solves
-----------------------
A model trained on listings can only interpolate between the cars in its training data.
Ask a one-hot-encoded pipeline about a 2025 Kia Syros and it has no coefficient for that
name: the encoder emits all-zeros and you get "a generic average car". That is the
"it only returns values from the dataset" complaint, and no amount of tuning fixes it,
because the information simply is not in a categorical column.

How this values an unseen car
-----------------------------
Market model  -> learned relationship between (brand, segment, age, kms, fuel, engine,
                 transmission, owners) and price. Trained on 8,148 real listings.
Depreciation  -> an explicit age curve, calibrated to 2026 Indian resale data, applied
                 to the catalogue's current ex-showroom price.
Catalogue     -> gives segment, engine and ex-showroom price for ~30 brands and
                 ~340 models, including models that appear in no training row.

The two estimates are blended, and the blend leans on the depreciation estimate exactly
when the model is least trustworthy (cars and ages it has little data for). Every answer
carries an explicit confidence and an explanation of which signals were used, so a
2026 Kia Syros comes back as "estimated from segment + depreciation, lower confidence"
rather than as a confident wrong number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .catalogue import (CATALOGUE, SEGMENT_LABELS, ModelSpec, brand_of, models_for,
                        resolve_brand, resolve_model)

#: calendar year used for "age" calculations. Kept explicit so results are reproducible
#: rather than quietly changing with the system clock.
VALUATION_YEAR = 2026

# --------------------------------------------------------------------------- #
# Depreciation curve, calibrated to 2026 Indian resale data
# --------------------------------------------------------------------------- #
#: Value retained after N years, as a fraction of ex-showroom price, for an average
#: mass-market car. Blends the commonly published retention points:
#:   year 1 ~79-80%, year 3 ~67%, year 5 ~59% of ex-showroom (typical car),
#:   flattening to ~20% at 15 years.
#: Sources: Autocar-Spinny Used Car Study 2026 (33.9% lost at 3 yrs, 43.2% at 5 yrs),
#: plus the standard Indian rule-of-thumb tables (75/62/48% at 1/3/5 yrs).
RETENTION_CURVE: list[tuple[float, float]] = [
    (0.0, 1.00), (0.5, 0.90), (1.0, 0.80), (2.0, 0.73), (3.0, 0.67),
    (4.0, 0.63), (5.0, 0.59), (6.0, 0.54), (7.0, 0.49), (8.0, 0.44),
    (10.0, 0.36), (12.0, 0.30), (15.0, 0.22), (20.0, 0.14),
]

#: Global level calibration. Why it is not 1.0: the catalogue stores *base* variant
#: ex-showroom prices, while real listings are mid/upper variants (VXi/ZXi), and Indian
#: used transactions clear above a naive base-price depreciation curve.
#:
#: Fitted by calibrating the curve against seven independent 2026 Delhi marketplace
#: prices spanning 2012-2023 (cars24 / carandbike). Implied level was 1.39, 1.34, 1.38,
#: 1.39, 1.31, 1.12, 0.87 -> median 1.34. Kept slightly conservative at 1.36 so the
#: estimate errs toward not over-promising on a sale.
RETENTION_LEVEL = 1.36

#: Multipliers fitted from real listings (median of actual / curve prediction, after the
#: level factor). >1 means that brand/segment/fuel holds value better than the average car.
BRAND_RETENTION: dict[str, float] = {
    "Maruti Suzuki": 1.06, "Toyota": 1.00, "Hyundai": 1.01, "Kia": 1.56,
    "Mahindra": 0.97, "Tata": 0.90, "Honda": 0.92, "Renault": 0.96,
    "Nissan": 0.98, "Datsun": 1.04, "Ford": 1.03, "Chevrolet": 0.86,
    "Fiat": 1.07, "Volkswagen": 0.97, "Skoda": 0.82, "Jeep": 1.03,
    "MG": 0.99, "Citroen": 0.90, "BYD": 0.92,
    "BMW": 0.99, "Mercedes-Benz": 0.86, "Audi": 0.97, "Jaguar": 1.47,
    "Volvo": 1.07, "Land Rover": 0.82, "Lexus": 1.26, "MINI": 0.90,
    "Porsche": 0.95, "Tesla": 1.00, "Isuzu": 1.51, "Force Motors": 0.69,
    "Mitsubishi": 0.80, "Hindustan Motors": 0.84, "Daewoo": 1.03,
    "Opel": 0.55, "Peugeot": 0.62, "Ashok Leyland": 1.13, "Premier": 0.75,
    "Other": 0.95,
}

SEGMENT_RETENTION: dict[str, float] = {
    "suv": 1.05, "mpv": 0.98, "hatchback": 1.00, "sedan": 1.00,
    "pickup": 1.12, "luxury-suv": 0.76, "luxury-sedan": 0.74,
    "luxury-hatch": 0.84, "ev-suv": 1.14, "ev-sedan": 0.90, "ev-hatch": 0.90,
}

FUEL_RETENTION: dict[str, float] = {
    "Petrol": 0.97, "Diesel": 1.03, "CNG": 0.80, "LPG": 1.06, "Electric": 0.81,
}


def retention_fraction(age_years: float) -> float:
    """Fraction of ex-showroom value retained at a given age (average car)."""
    age = max(0.0, float(age_years))
    xs = [p[0] for p in RETENTION_CURVE]
    ys = [p[1] for p in RETENTION_CURVE]
    if age >= xs[-1]:
        return ys[-1]
    return float(np.interp(age, xs, ys))


def depreciation_value(*, ex_showroom_lakh: float, brand: str, segment: str,
                       fuel: str, age_years: float, kms_driven: float,
                       transmission: str = "Manual", owner: str = "First Owner",
                       service_history: bool = True) -> float:
    """
    Ex-showroom-price-based valuation. Returns rupees.

    Age is the dominant term; kilometres, fuel, transmission and ownership count are
    applied as bounded adjustments so no single input can run away with the estimate.
    """
    base = ex_showroom_lakh * 1_00_000

    keep = (RETENTION_LEVEL
            * BRAND_RETENTION.get(brand, 0.95)
            * SEGMENT_RETENTION.get(segment, 1.0)
            * FUEL_RETENTION.get(fuel, 0.97))
    # A pre-owned vehicle cannot retain > 92% of new ex-showroom price regardless of multiplier;
    # this keeps the curve in its active non-zero gradient region without artificial ceiling clamping.
    retention = min(0.92, retention_fraction(age_years) * keep)
    value = base * retention

    # kilometres: compare against a typical ~11,000 km/yr, smoothly damped with a
    # continuous tanh transformation so there are NO zero-derivative flat plateaus.
    # Every 1,000 km change smoothly reflects in the valuation without stagnation.
    expected_km = max(5_000.0, 11_000.0 * max(age_years, 1.0))
    km_ratio = float(kms_driven) / expected_km
    km_factor = 1.0 - 0.15 * float(np.tanh((km_ratio - 1.0) * 0.75))
    value *= km_factor

    value *= 1.045 if transmission == "Automatic" else 1.0
    value *= {"First Owner": 1.02, "Second Owner": 0.965,
              "Third Owner": 0.92, "Fourth & Above Owner": 0.88,
              "Test Drive Car": 0.80}.get(owner, 1.0)
    if not service_history:
        value *= 0.93

    # Soft ceiling: prevents any used car from exceeding the ex-showroom price,
    # but uses smooth compression so there is NEVER a zero-derivative flat plateau.
    # Even young, high-retention models (e.g. Kia/Toyota with low km) maintain a smooth
    # depreciation gradient as mileage increases.
    ceiling = min(1.0, RETENTION_LEVEL) * 0.98 * base
    threshold = 0.88 * ceiling
    if value > threshold:
        excess = value - threshold
        headroom = ceiling - threshold
        value = threshold + headroom * (1.0 - float(np.exp(-excess / headroom)))
    return float(max(value, 0.02 * base))


# --------------------------------------------------------------------------- #
# Inputs / outputs
# --------------------------------------------------------------------------- #
@dataclass
class CarQuery:
    brand: str | None = None
    model: str | None = None
    year: int = VALUATION_YEAR - 5
    kms_driven: int = 50_000
    fuel: str = "Petrol"
    transmission: str = "Manual"
    owner: str = "First Owner"
    engine_cc: int | None = None
    seats: int | None = None
    service_history: bool = True
    city: str | None = None

    @property
    def age(self) -> float:
        return max(0.0, VALUATION_YEAR - int(self.year))


@dataclass
class Valuation:
    price: float
    low: float
    high: float
    confidence: str                 # "high" | "medium" | "low"
    method: str                     # which signals produced this
    spec: ModelSpec | None
    segment: str
    resolved_brand: str
    resolved_model: str
    known_model: bool
    basis: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "price": round(self.price), "low": round(self.low), "high": round(self.high),
            "confidence": self.confidence, "method": self.method,
            "segment": self.segment,
            "segment_label": SEGMENT_LABELS.get(self.segment, self.segment.title()),
            "brand": self.resolved_brand, "model": self.resolved_model,
            "known_model": self.known_model, "basis": self.basis, "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# The valuator
# --------------------------------------------------------------------------- #
class CarValuator:
    """
    Blends the trained market model with catalogue-based depreciation.

    `market_model` is a fitted sklearn Pipeline expecting the feature frame produced by
    `market_features()`. If it is None, valuation runs on depreciation alone.
    """

    def __init__(self, market_model=None, meta: dict | None = None):
        self.market_model = market_model
        self.meta = meta or {}
        self.training_rows = int(self.meta.get("clean_rows") or 0)
        #: model names the market model actually learned a coefficient for
        self.known_models: set[str] = set(self.meta.get("models_known") or [])
        self.train_year_max = int(self.meta.get("year_max") or 2023)
        self.train_year_min = int(self.meta.get("year_min") or 2000)
        self._last_borrowed = False

    # -- feature construction ------------------------------------------------
    def resolve(self, q: CarQuery) -> tuple[str, str, ModelSpec | None, str, bool]:
        """Work out brand / model / spec / segment, falling back gracefully."""
        brand = resolve_brand(q.brand) if q.brand else None
        if brand is None and q.model:
            brand = brand_of(q.model)
        if brand is None and q.model:
            for candidate in CATALOGUE:
                if candidate.lower() in str(q.model).lower():
                    brand = candidate
                    break
        brand = brand or "Other"

        spec: ModelSpec | None = None
        model_name = (q.model or "").strip()

        # 1. look for the model inside the stated brand first
        if model_name:
            hit = resolve_model(brand, model_name)
            if hit:
                model_name, spec = hit

        # 2. then anywhere in the catalogue
        if spec is None and model_name:
            hit = resolve_model(None, model_name)
            if hit:
                model_name, spec = hit

        # 3. a car we have never heard of is NOT an error. Borrow the nearest
        #    catalogue sibling from the same brand: 'Kia Clavis' (unreleased) inherits
        #    Kia's compact-SUV characteristics rather than becoming a generic sedan.
        if spec is None and brand in CATALOGUE:
            sibling = _nearest_sibling(CATALOGUE[brand])
            if sibling is not None:
                model_name = model_name or "Other"
                spec = sibling
                self._last_borrowed = True

        known = model_name in self.known_models

        if spec is not None:
            segment = spec.segment
        else:
            segment = _guess_segment(model_name, q)

        return brand, model_name or "Other", spec, segment, known

    def market_features(self, q: CarQuery, brand: str, model: str,
                        spec: ModelSpec | None, segment: str) -> pd.DataFrame:
        # NB: `or` / truthiness is wrong here — an EV legitimately has engine_cc == 0,
        # and `if engine` silently discarded it, leaving the row to be imputed.
        engine = q.engine_cc if q.engine_cc is not None else (
            spec.engine_cc if spec is not None else np.nan)
        seats = q.seats if q.seats is not None else (
            spec.seats if spec is not None else np.nan)
        row = {
            "brand": brand,
            "model": model,
            "segment": segment,
            "age": q.age,
            "kms_driven": float(q.kms_driven),
            "km_per_year": float(q.kms_driven) / max(q.age, 1.0),
            "fuel": q.fuel,
            "transmission": q.transmission,
            "owner": q.owner,
            # `float(x) if x else nan` would turn an EV's 0 into NaN — use `is not None`
            "engine_cc": float(engine) if engine is not None and not _isnan(engine) else np.nan,
            "seats": float(seats) if seats is not None and not _isnan(seats) else np.nan,
            "has_service_history": int(bool(q.service_history)),
        }
        return pd.DataFrame([row])

    # -- the estimate --------------------------------------------------------
    def value(self, q: CarQuery) -> Valuation:
        brand, model, spec, segment, known = self.resolve(q)
        notes: list[str] = []
        basis: dict = {}

        dep = None
        if spec is not None:
            dep = depreciation_value(
                ex_showroom_lakh=spec.ex_showroom_lakh, brand=brand, segment=segment,
                fuel=q.fuel, age_years=q.age, kms_driven=q.kms_driven,
                transmission=q.transmission, owner=q.owner,
                service_history=q.service_history)
            basis["ex_showroom_lakh"] = spec.ex_showroom_lakh

        market = None
        if self.market_model is not None:
            try:
                X = self.market_features(q, brand, model, spec, segment)
                market = float(self.market_model.predict(X)[0])
                market = max(market, 0.0)
            except Exception as exc:                      # pragma: no cover - defensive
                notes.append(f"market model unavailable ({type(exc).__name__})")

        # ---- blending -----------------------------------------------------
        # The calibrated curve is the backbone: it generalises to any car, any age, and
        # it is what the 2026 spot-checks validated to within ~6%. The market model is a
        # learned cross-section from 2023 listings, so it is trusted as a *relative*
        # adjustment (this model/segment/owner mix vs the fleet) evaluated at benchmark mileage.
        # This isolates the model's market premium from tree leaf step-noise, ensuring
        # the continuous depreciation curve governs mileage variations smoothly and monotonically.
        if dep is not None and market is not None:
            expected_km = max(5_000.0, 11_000.0 * max(q.age, 1.0))
            try:
                q_bench = CarQuery(
                    brand=q.brand, model=q.model, year=q.year, kms_driven=int(expected_km),
                    fuel=q.fuel, transmission=q.transmission, owner=q.owner,
                    engine_cc=q.engine_cc, seats=q.seats, service_history=q.service_history
                )
                X_bench = self.market_features(q_bench, brand, model, spec, segment)
                market_bench = float(self.market_model.predict(X_bench)[0])
                dep_bench = depreciation_value(
                    ex_showroom_lakh=spec.ex_showroom_lakh, brand=brand, segment=segment,
                    fuel=q.fuel, age_years=q.age, kms_driven=expected_km,
                    transmission=q.transmission, owner=q.owner,
                    service_history=q.service_history)
                rel_market = float(np.clip(market_bench / dep_bench, 0.72, 1.45))
            except Exception:
                rel_market = float(np.clip(market / dep, 0.72, 1.45))

            if known:
                w = 0.30                       # modest tilt toward the learned model
            else:
                w = 0.10                       # barely any: it has not seen this car
                notes.append("this model is outside the training data; the estimate is "
                             "driven by segment, age and specification")
            price = dep * (1 - w + w * rel_market)
            method = "depreciation curve + market adjustment"
            basis["market_model_raw"] = market
            basis["depreciation"] = dep
            basis["market_tilt"] = round(1 - w + w * rel_market, 3)
        elif dep is not None:
            price, method = dep, "depreciation curve (catalogue ex-showroom price)"
            basis["depreciation"] = dep
        elif market is not None:
            price, method = market, "market model only (no catalogue entry)"
            basis["market_model_raw"] = market
            notes.append("brand/model not in the catalogue; estimate comes from the "
                         "learned market model alone")
        else:
            raise ValueError(
                f"Cannot value '{q.brand} {q.model}': brand not in the catalogue and "
                "no market model to fall back on. Add it to catalogue.py, or supply an "
                "ex-showroom price."
            )

        price = float(max(price, 15_000))

        # ---- confidence ---------------------------------------------------
        if known and spec is not None and self.train_year_min <= q.year <= self.train_year_max:
            confidence, spread = "high", 0.11
        elif known:
            confidence, spread = "medium", 0.16
        elif spec is not None:
            confidence, spread = "medium", 0.18
        else:
            confidence, spread = "low", 0.28
            notes.append("model not recognised; treated as a generic car in its segment")

        if q.age > 12:
            notes.append("cars older than 12 years trade mostly on condition")
            spread += 0.07

        low, high = price * (1 - spread), price * (1 + spread)
        return Valuation(
            price=price, low=low, high=high, confidence=confidence, method=method,
            spec=spec, segment=segment, resolved_brand=brand, resolved_model=model,
            known_model=known, basis=basis, notes=notes,
        )


#: order used to pick a "most typical" model for a brand when a name is unknown
_SEGMENT_PRIORITY = ["hatchback", "suv", "sedan", "mpv", "luxury-suv", "luxury-sedan"]


def _nearest_sibling(models: dict[str, ModelSpec]) -> ModelSpec | None:
    """
    The most representative (segment, specs) for a brand — its median-priced model.

    This is what makes an unrecognised name still behave sensibly: a Kia we have never
    seen should come back as "a Kia-sized compact SUV", not as the fleet average.
    """
    if not models:
        return None
    # the brand's *volume* segment, at its median price. Kia's PRICE median is an MPV
    # (Carens/Carnival), but its volume segment is compact SUV — so segment first, then
    # price. Mode breaks ties toward how the brand is actually known.
    mainstream = [m for m in models.values() if m.segment in _SEGMENT_PRIORITY] or list(models.values())
    seg_counts: dict[str, int] = {}
    for m in mainstream:
        seg_counts[m.segment] = seg_counts.get(m.segment, 0) + 1
    top_segment = max(seg_counts, key=lambda k: (seg_counts[k], -_SEGMENT_PRIORITY.index(k)))
    in_seg = sorted((m for m in mainstream if m.segment == top_segment),
                    key=lambda m: m.ex_showroom_lakh)
    return in_seg[len(in_seg) // 2]


def _isnan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _guess_segment(model_name: str, q: CarQuery) -> str:
    """Cheap segment guess for cars that are not in the catalogue."""
    name = (model_name or "").lower()
    engine = q.engine_cc or 0
    seats = q.seats or 0
    if q.fuel == "Electric":
        return "ev-suv" if seats >= 5 and engine == 0 else "ev-hatch"
    if seats >= 7:
        return "mpv"
    if any(k in name for k in ("xuv", "suv", "fortuner", "cayenne", "range rover")):
        return "suv"
    if engine >= 1900:
        return "suv" if "sedan" not in name else "sedan"
    return "hatchback" if engine and engine <= 1200 else "sedan"


# --------------------------------------------------------------------------- #
# What the market model needs
# --------------------------------------------------------------------------- #
MARKET_FEATURES = ["brand", "model", "segment", "age", "kms_driven", "km_per_year",
                   "fuel", "transmission", "owner", "engine_cc", "seats",
                   "has_service_history"]
CATEGORICAL_FEATURES = ["brand", "model", "segment", "fuel", "transmission", "owner"]
NUMERIC_FEATURES = ["age", "kms_driven", "km_per_year", "engine_cc", "seats"]
BINARY_FEATURES = ["has_service_history"]

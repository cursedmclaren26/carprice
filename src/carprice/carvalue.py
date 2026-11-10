"""
carvalue.py
-----------
Train the market model and value any car from the command line.

    python -m carprice.carvalue train
    python -m carprice.carvalue price --brand "Maruti Suzuki" --model Swift --year 2019 --km 50000
    python -m carprice.carvalue price --brand Kia --model Syros --year 2025 --km 6000
    python -m carprice.carvalue brands
    python -m carprice.carvalue models --brand Hyundai
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _project_root() -> Path:
    """
    Locate the project root for default paths.

    Two layouts must both work: a source checkout (`<root>/src/carprice/carvalue.py`)
    and an editable install (the module resolves through site-packages). Walking up
    from the file is right for the first; for the second, fall back to the current
    working directory. Getting this wrong writes model files into somebody else's
    checkout — which is exactly what happened during testing.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


ROOT = _project_root()
DEFAULT_RAW = ROOT / "data" / "raw" / "cardekho_used_cars.csv"
DEFAULT_MODEL = ROOT / "models" / "market_model.pkl"


def _inr(value: float) -> str:
    """Indian digit grouping: 479105 -> '4,79,105'."""
    s = str(int(round(abs(value))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if value < 0 else "") + s


def cmd_train(args) -> int:
    from .market import load_market_raw, prepare_market, save_market, train_market

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"Raw dataset not found: {raw_path}")
        print("Copy a CarDekho-style CSV there, or run:  python -m carprice.carvalue fetch")
        return 1

    raw = load_market_raw(raw_path)
    print(f"raw rows: {len(raw):,}")
    features, audit = prepare_market(raw)
    for step, rows in audit.itertuples(index=False):
        print(f"  {step:<48} {rows:>6,}")

    print("\ntraining (cross-validation is grouped by model name, so a fold never\n"
          "validates on a car whose near-identical siblings it just trained on):\n")
    model, meta, _ = train_market(features)

    save_market(model, meta, args.out)
    features.to_csv(Path(args.out).parent.parent / "data" / "market_features.csv",
                    index=False)
    print(f"\nsaved {args.out}")
    print(f"saved {Path(args.out).with_suffix('.json')}")
    return 0


def cmd_price(args) -> int:
    from .market import load_market
    from .valuation import CarQuery, CarValuator

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"No trained model at {model_path}. Run:  python -m carprice.carvalue train")
        return 1
    model, meta = load_market(model_path)
    valuator = CarValuator(model, meta)

    query = CarQuery(brand=args.brand, model=args.model, year=args.year,
                     kms_driven=args.km, fuel=args.fuel,
                     transmission=args.transmission, owner=args.owner,
                     service_history=not args.no_service_history)
    v = valuator.value(query)

    if args.json:
        print(json.dumps(v.to_dict(), indent=2))
        return 0

    print(f"\n  {args.year} {v.resolved_brand} {v.resolved_model}"
          f"  ({args.km:,} km, {args.fuel}, {args.transmission})")
    print(f"  {'-' * 56}")
    print(f"  Estimated selling price   Rs {_inr(v.price)}")
    print(f"  Likely range              Rs {_inr(v.low)} - {_inr(v.high)}")
    print(f"  Confidence                {v.confidence}")
    print(f"  Method                    {v.method}")
    if not v.known_model:
        print(f"  Note                      this exact model is not in the training "
              f"data;\n                            valued from segment, age and specs")
    for note in v.notes:
        print(f"  ! {note}")
    print()
    return 0


def cmd_brands(args) -> int:
    from .catalogue import all_brands
    for b in all_brands():
        print(b)
    return 0


def cmd_models(args) -> int:
    from .catalogue import models_for
    for m in models_for(args.brand):
        print(m)
    return 0


def cmd_fetch(args) -> int:
    """Download the public CarDekho dump into data/raw/."""
    import urllib.request

    url = ("https://raw.githubusercontent.com/pragyapathak04/Cars-Data/main/"
           "Car%20dekho.csv")
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"saved {dest}  ({dest.stat().st_size / 1024:.0f} KB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="carvalue", description="Value any used car, including unseen models.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the market model")
    t.add_argument("--raw", default=str(DEFAULT_RAW))
    t.add_argument("--out", default=str(DEFAULT_MODEL))
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("price", help="value one car")
    p.add_argument("--brand", required=True, help='e.g. "Maruti Suzuki", Hyundai, Kia')
    p.add_argument("--model", required=True, help="e.g. Swift, Creta, Syros")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--km", type=int, required=True, dest="km",
                   help="kilometres driven")
    p.add_argument("--fuel", default="Petrol",
                   choices=["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    p.add_argument("--transmission", default="Manual", choices=["Manual", "Automatic"])
    p.add_argument("--owner", default="First Owner",
                   choices=["First Owner", "Second Owner", "Third Owner",
                            "Fourth & Above Owner", "Test Drive Car"])
    p.add_argument("--no-service-history", action="store_true")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL))
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_price)

    b = sub.add_parser("brands", help="list known brands")
    b.set_defaults(func=cmd_brands)

    m = sub.add_parser("models", help="list the models known for a brand")
    m.add_argument("--brand", required=True)
    m.set_defaults(func=cmd_models)

    f = sub.add_parser("fetch", help="download the public CarDekho dataset")
    f.add_argument("--out", default=str(DEFAULT_RAW))
    f.set_defaults(func=cmd_fetch)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

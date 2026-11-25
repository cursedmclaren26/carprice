"""
app.py
------
Car valuation web app.

Unlike the tutorial's form, the model field is **free text with suggestions**: you can
type a car that appears in no dataset (a 2026 Kia Clavis, a Tesla) and still get a price,
because valuation falls back to segment + specification rather than to a lookup table.

Supports both standard form submission (`POST /`) and live dynamic AJAX updates
(`POST /api/predict`) aligned with the 9 reduced ML features (7 user inputs +
2 catalogue-resolved physical specs: segment & engine_cc).

Run:
    python web/app.py                 # http://127.0.0.1:5000
    python web/app.py --port 5001
"""
from __future__ import annotations

import sys
from pathlib import Path

_CORE = ("numpy", "pandas", "sklearn", "joblib", "flask")
_MISSING = []
for _mod in _CORE:
    try:
        __import__(_mod)
    except ModuleNotFoundError:
        _MISSING.append(_mod)

if _MISSING:
    _display = {"sklearn": "scikit-learn", "numpy": "numpy", "pandas": "pandas",
                "joblib": "joblib", "flask": "flask"}
    _names = ", ".join(_display[m] for m in _MISSING)
    sys.exit(
        f"\nMissing dependencies: {_names}\n\n"
        "  The web app needs the core stack (numpy, pandas, scikit-learn, joblib)\n"
        "  plus Flask. A partial install is the usual cause.\n\n"
        "  From the carprice folder, install everything the app needs:\n\n"
        "      pip install -e \".[eda,web]\"\n\n"
        "  or, if the package is already installed and you only need the extras:\n\n"
        "      pip install flask numpy pandas scikit-learn joblib\n\n"
        "  Quote the brackets. In PowerShell and zsh, unquoted .[eda,web] is parsed\n"
        "  as a glob and the command fails before pip ever runs.\n"
    )

from flask import Flask, jsonify, render_template, request  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from carprice.catalogue import CATALOGUE, SEGMENT_LABELS  # noqa: E402
from carprice.market import load_market  # noqa: E402
from carprice.valuation import (MARKET_FEATURES, CarQuery, CarValuator,  # noqa: E402
                                VALUATION_YEAR)

MODEL_PATH = ROOT / "models" / "market_model.pkl"

app = Flask(__name__, template_folder=str(ROOT / "templates"),
            static_folder=str(ROOT / "static"))

VALUATOR: CarValuator | None = None
META: dict = {}
LOAD_ERROR: str | None = None

try:
    _model, META = load_market(MODEL_PATH)
    VALUATOR = CarValuator(_model, META)
except Exception as exc:  # pragma: no cover - fresh checkout
    LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def _inr(value: float) -> str:
    """479105 -> '4,79,105' (Indian digit grouping)."""
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


def _suggestions() -> dict[str, list[str]]:
    return {brand: sorted(models) for brand, models in CATALOGUE.items()}


def _parse_form_payload(src) -> dict:
    return {
        "brand": str(src.get("brand", "") or ""),
        "model": str(src.get("model", "") or ""),
        "year": str(src.get("year", str(VALUATION_YEAR - 5)) or str(VALUATION_YEAR - 5)),
        "kms_driven": str(src.get("kms_driven", "50000") or "50000"),
        "fuel": str(src.get("fuel", "Petrol") or "Petrol"),
        "transmission": str(src.get("transmission", "Manual") or "Manual"),
        "owner": str(src.get("owner", "First Owner") or "First Owner"),
    }


def _evaluate_query(form: dict) -> dict:
    if VALUATOR is None:
        raise RuntimeError(LOAD_ERROR or "Model not loaded")
    try:
        year = int(str(form["year"]).strip())
    except ValueError as exc:
        raise ValueError("Enter a valid numeric year.") from exc
    try:
        kms = int(float(str(form["kms_driven"]).strip()))
    except ValueError as exc:
        raise ValueError("Enter a valid numeric odometer reading.") from exc

    if not 1990 <= year <= VALUATION_YEAR:
        raise ValueError(f"Year must be between 1990 and {VALUATION_YEAR}.")
    if not 0 <= kms <= 600_000:
        raise ValueError("Kilometres driven must be between 0 and 6,00,000.")
    if not form["model"].strip():
        raise ValueError("Enter the model — e.g. Swift, Creta, Syros.")

    query = CarQuery(
        brand=form["brand"].strip() or None,
        model=form["model"].strip(),
        year=year,
        kms_driven=kms,
        fuel=form["fuel"],
        transmission=form["transmission"],
        owner=form["owner"],
    )
    v = VALUATOR.value(query)
    result = v.to_dict()
    result["price_text"] = _inr(v.price)
    result["low_text"] = _inr(v.low)
    result["high_text"] = _inr(v.high)
    result["engine_cc"] = int(v.spec.engine_cc) if v.spec is not None else None
    result["features_count"] = len(MARKET_FEATURES)
    return result


@app.route("/", methods=["GET", "POST"])
@app.route("/docs", methods=["GET"])
def index():
    initial_view = "docs" if request.path == "/docs" else "predictor"
    if LOAD_ERROR:
        return render_template("index.html", load_error=LOAD_ERROR, result=None,
                               form={}, brands=[], this_year=VALUATION_YEAR,
                               names_by_brand={}, segments=SEGMENT_LABELS,
                               meta=META, initial_view=initial_view)

    form = _parse_form_payload(request.form)
    result, error = None, None

    if request.method == "POST":
        try:
            result = _evaluate_query(form)
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        brands=sorted(CATALOGUE), names_by_brand=_suggestions(),
        this_year=VALUATION_YEAR, form=form, result=result, error=error,
        meta=META, load_error=None, segments=SEGMENT_LABELS,
        initial_view=initial_view,
    )


@app.post("/api/predict")
def api_predict():
    """Real-time JSON valuation endpoint for dynamic frontend updates."""
    if LOAD_ERROR:
        return jsonify({"error": LOAD_ERROR}), 503
    payload = request.get_json(silent=True) or request.form or {}
    form = _parse_form_payload(payload)
    try:
        result = _evaluate_query(form)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/health")
def health():
    tm = META.get("test_metrics", {})
    return jsonify({
        "status": "ok" if VALUATOR else "model_missing",
        "market_model": META.get("market_model"),
        "features_count": META.get("features_count", len(MARKET_FEATURES)),
        "features": META.get("features", MARKET_FEATURES),
        "rows": META.get("rows"),
        "cv_r2": tm.get("cv_r2"),
        "test_r2": tm.get("test_r2"),
        "test_mae": tm.get("test_mae"),
        "test_mape": tm.get("test_mape"),
    })


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run the car valuation app.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if LOAD_ERROR:
        print(f"\n[!] No model loaded: {LOAD_ERROR}")
        print("    Train one first:  python -m carprice.carvalue train\n")

    app.run(host=args.host, port=args.port, debug=args.debug)

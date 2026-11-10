"""
Used-car valuation for the Indian market.

Public API:
    from carprice.valuation import CarQuery, CarValuator
    from carprice.carvalue import main          # CLI entry point

Imports here are **lazy**. An eager `from .market import ...` would drag numpy,
pandas and scikit-learn into the interpreter the moment anything touched the
package — even `import carprice.catalogue`, whose only needs are `dataclasses` and
`re`. That is how a dict lookup ends up failing with "No module named 'numpy'".
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "2.0.0"

#: name -> submodule that provides it
_EXPORTS = {
    "CATALOGUE": "catalogue",
    "SEGMENT_LABELS": "catalogue",
    "resolve_brand": "catalogue",
    "resolve_model": "catalogue",
    "load_market_raw": "market",
    "prepare_market": "market",
    "train_market": "market",
    "CarQuery": "valuation",
    "CarValuator": "valuation",
    "Valuation": "valuation",
    "depreciation_value": "valuation",
}

__all__ = [*_EXPORTS, "__version__"]

if TYPE_CHECKING:  # pragma: no cover - for editors and type checkers only
    from .catalogue import CATALOGUE, SEGMENT_LABELS, resolve_brand, resolve_model
    from .market import load_market_raw, prepare_market, train_market
    from .valuation import CarQuery, CarValuator, Valuation, depreciation_value


def __getattr__(name: str):
    """Resolve module-level names on first use (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value          # cache, so the lookup happens once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)

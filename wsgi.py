"""
wsgi.py
-------
Production WSGI entry point for the Car Valuation app.

Gunicorn / Render starts this with:
    gunicorn wsgi:app

The file lives at the project root so that ROOT in app.py resolves to this same
directory (parent of web/), which is where models/ and templates/ live.
"""
import sys
from pathlib import Path

# Make `carprice` importable whether the package is installed (pip install -e .)
# or the src layout is used directly on a fresh Render build.
_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Import app from web/app.py by adding web/ to path and importing directly.
# We do NOT use `from web.app import app` because web/ has no __init__.py —
# it was deliberately kept as a plain script directory, not a package.
_web = Path(__file__).resolve().parent / "web"
if str(_web) not in sys.path:
    sys.path.insert(0, str(_web))

from app import app  # noqa: E402  (app.py sets ROOT = parent of web/ = project root)

if __name__ == "__main__":
    app.run()


"""
check_setup.py
--------------
One command that tells you what is missing and exactly how to fix it.

    python check_setup.py

Run it from the `carprice` folder, with whatever interpreter you intend to use.
If you have several Pythons installed, this is the fastest way to find out which
one actually has the packages.

Exit code 0 = ready to run. 1 = something to fix.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"

CORE = [
    ("numpy", "numpy", "numpy"),
    ("pandas", "pandas", "pandas"),
    ("joblib", "joblib", "joblib"),
    ("sklearn", "scikit-learn", "scikit-learn"),
]
OPTIONAL = [
    ("flask", "flask", "Flask — needed by the web app"),
    ("matplotlib", "matplotlib", "matplotlib — needed by the notebook"),
    ("seaborn", "seaborn", "seaborn — needed by the notebook"),
]
MODEL_FILES = ["models/market_model.pkl", "models/market_model.json"]
DATA_FILES = ["data/raw/cardekho_used_cars.csv"]

ISSUES: list[str] = []


def check_import(module: str, pip_name: str) -> str | None:
    """Return the installed version, or None if the import would fail."""
    spec = importlib.util.find_spec(module)
    if spec is None:
        return None
    try:
        import importlib.metadata as md
        try:
            return md.version(pip_name)
        except md.PackageNotFoundError:
            return "installed"
    except Exception:
        return "installed (version unknown)"


def main() -> int:
    here = Path(__file__).resolve().parent
    in_project = (here / "pyproject.toml").exists()

    print("carprice — setup check")
    print("=" * 62)
    print(f"python            {sys.version.split()[0]}")
    print(f"executable        {sys.executable}")
    print(f"project folder    {here}")
    if not in_project:
        print(f"                  {WARN} this does not look like the carprice folder")
    print()

    # ---- core ----
    print("core dependencies")
    missing_core = []
    for module, pip_name, _ in CORE:
        version = check_import(module, pip_name)
        if version:
            print(f"  {OK} {pip_name:<15} {version}")
        else:
            print(f"  {BAD} {pip_name:<15} missing")
            missing_core.append(pip_name)
    print()

    # ---- optional ----
    print("optional")
    missing_web, missing_eda = [], []
    for module, pip_name, note in OPTIONAL:
        version = check_import(module, pip_name)
        if version:
            print(f"  {OK} {pip_name:<15} {version}")
        else:
            print(f"  {WARN} {pip_name:<15} missing — {note}")
            (missing_web if module == "flask" else missing_eda).append(pip_name)
    print()

    # ---- package itself ----
    print("the carprice package")
    spec = importlib.util.find_spec("carprice")
    if spec is None:
        print(f"  {BAD} not importable — run:  pip install -e .")
        ISSUES.append("install the package: pip install -e .")
    else:
        try:
            import carprice
            print(f"  {OK} carprice {carprice.__version__}")
            print(f"        from {Path(carprice.__file__).parent}")
            source = here / "src"
            if source.exists() and str(source) not in str(Path(carprice.__file__)):
                print(f"  {WARN} imported from a different copy than {source}")
                print(f"        -> you may be running stale code; "
                      f"re-run: pip install -e .")
        except Exception as exc:
            print(f"  {BAD} import failed: {type(exc).__name__}: {exc}")
            ISSUES.append("re-import failing; reinstall with: pip install -e .")
    print()

    # ---- model ----
    print("model and data")
    model_ok = True
    for rel in MODEL_FILES:
        path = here / rel
        if path.exists():
            print(f"  {OK} {rel:<34} {path.stat().st_size / 1024:>6.0f} KB")
        else:
            print(f"  {BAD} {rel:<34} missing")
            model_ok = False
    if not model_ok:
        ISSUES.append("train the model: python -m carprice.carvalue train")

    for rel in DATA_FILES:
        path = here / rel
        if path.exists():
            print(f"  {OK} {rel:<34} {path.stat().st_size / 1024:>6.0f} KB")
        else:
            print(f"  {WARN} {rel:<34} missing")
            ISSUES.append("download the dataset: python -m carprice.carvalue fetch")
    print()

    # ---- verdict ----
    print("=" * 62)
    if missing_core or missing_web or missing_eda:
        packages = missing_core + missing_web + missing_eda
        print("Install everything at once (note the quotes):")
        print()
        print('    pip install -e ".[eda,web]"')
        print()
        print("Or just the missing packages:")
        print(f"    pip install {' '.join(packages)}")
        print()
        if missing_core:
            ISSUES.insert(0, f"missing core packages: {', '.join(missing_core)}")

    if ISSUES:
        print("to fix:")
        for issue in ISSUES:
            print(f"  - {issue}")
        print()
        print("Then run this check again.")
        return 1

    print("Ready. Start the web app with:")
    print()
    print(f"    {Path(sys.executable).name} web/app.py")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

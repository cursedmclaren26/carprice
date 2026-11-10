# Running it manually after installing

You've done `pip install -e ".[eda,web]"`. Here is everything from that point on.

There are **two ways to use it**: the web app (a form in a browser) and the CLI (one
command per car). Both read `models/market_model.pkl`. If that file is missing, both
tell you to train first.

---

## 0. Where you must be standing

**The CLI does not care.** `carvalue` and `python -m carprice.carvalue` work from any
directory once installed — tested from `/tmp`.

**The web app needs the project folder** it lives in, because it loads `models/`,
`templates/` and `static/` by path relative to itself. It resolves those absolutely, so
you can launch it from anywhere *as long as you give the full path*:

```bash
python /path/to/carprice/web/app.py          # works from any directory
```

But the simplest habit — and what the rest of this doc assumes — is:

```bash
cd carprice
```

---

## 1. First run on a fresh machine

### Check what you have first

```bash
python check_setup.py
```

It prints one line per dependency, then exactly what to run if anything is missing.
Run it with **the same interpreter you intend to use** — if you have several Pythons,
this is the fastest way to find out which one has the packages.

### Install the extras, not just the package

`pip install -e .` installs **only the core**: pandas, scikit-learn, numpy, joblib.
Flask, matplotlib and seaborn are optional extras, so a bare install leaves the web app
unable to start.

| what you want to do | install command |
|---|---|
| web app | `pip install -e ".[web]"` |
| notebook / charts | `pip install -e ".[eda]"` |
| both, plus everything | `pip install -e ".[eda,web]"` |
| just Flask, on its own | `pip install flask` |

**Quote the brackets.** In PowerShell and zsh, an unquoted `.[eda,web]` is parsed as a
glob and the command fails before pip runs:

```powershell
pip install -e ".[eda,web]"     # correct
pip install -e .[eda,web]       # PowerShell: "The term '.[eda,web]' is not recognized"
```

If you already ran a bare `pip install -e .`, you do not need to reinstall anything
else — just add Flask:

```powershell
pip install flask
```

### Get the model and data

Installing the package does **not** install the model or the dataset. Three steps, once:

```bash
cd carprice
carvalue fetch      # downloads the 1 MB dataset into data/raw/
carvalue train      # fits the model, ~25 s, writes models/market_model.pkl
```

Then either interface works.

> If you copied the `carprice` folder from somewhere else, `models/` may already be
> populated — check with `ls models/`. If `market_model.pkl` and `.json` are both there,
> skip straight to step 2.

---

## 2. The web app

```bash
cd carprice
python web/app.py
```

```
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

Open **<http://127.0.0.1:5000>** in a browser. Ctrl-C in the terminal stops it.

### Which command works on your machine

| your situation | command |
|---|---|
| Linux / macOS, venv active | `python web/app.py` |
| Linux / macOS, no venv | `python3 web/app.py` |
| Windows cmd / PowerShell | `python web\app.py` |
| `python` not found | try `python3`, then `py` (Windows) |

`python3` vs `python` is the single most common stumble. If one fails with
*command not found*, use the other. On any modern Linux/macOS venv both work.

### Options

```bash
python web/app.py --port 5001      # port 5000 taken (macOS AirPlay, another app)
python web/app.py --host 127.0.0.1 # local-only, don't expose on your LAN
python web/app.py --debug          # auto-reload while editing code
```

**Windows note:** `--host 127.0.0.1` is worth using, otherwise Windows Firewall will pop
up asking whether to allow Python on private networks. Allowing it is fine for a local
demo.

**macOS note:** port 5000 is taken by AirPlay Receiver by default. Either use
`--port 5001` or turn off *System Settings → General → AirDrop & Handoff → AirPlay
Receiver*.

---

## 3. The CLI

After `pip install -e .`, the `carvalue` command is on your PATH:

```bash
carvalue price --brand "Maruti Suzuki" --model Swift --year 2019 --km 50057
```

If `carvalue` is **not found** (PATH problems are common on Windows and with some venv
setups), the module form always works because it does not depend on PATH:

```bash
python -m carprice.carvalue price --brand "Maruti Suzuki" --model Swift --year 2019 --km 50057
```

Both produce:

```
  2019 Maruti Suzuki Swift  (50,057 km, Petrol, Manual)
  --------------------------------------------------------
  Estimated selling price   Rs 4,91,831
  Likely range              Rs 4,37,729 - 5,45,932
  Confidence                high
  Method                    depreciation curve + market adjustment
```

### All the subcommands

```bash
carvalue fetch                              # download the dataset
carvalue train                              # fit + save the model
carvalue price --brand Kia --model Syros --year 2025 --km 6000
carvalue brands                             # 38 known brands
carvalue models --brand Hyundai             # models for one brand
```

### All the `price` flags

```bash
carvalue price \
  --brand "Maruti Suzuki" \      # required — quote names with spaces
  --model Swift \                # required — free text, need not be in the data
  --year 2019 \                  # required — 1990..2026
  --km 50057 \                   # required — kilometres driven
  --fuel Petrol \                # Petrol | Diesel | CNG | LPG | Electric
  --transmission Manual \        # Manual | Automatic
  --owner "First Owner" \        # First Owner | Second Owner | Third Owner | Fourth & Above Owner
  --no-service-history \         # optional flag — drops price ~7%
  --json                         # machine-readable output
```

**Quoting matters.** `--owner "First Owner"` and `--brand "Maruti Suzuki"` need quotes
because of the space. This is a shell rule, not a program one.

---

## 4. Using it from Python

```python
from carprice.valuation import CarQuery, CarValuator
from carprice.market import load_market

model, meta = load_market("models/market_model.pkl")
valuator = CarValuator(model, meta)

v = valuator.value(CarQuery(
    brand="Maruti Suzuki", model="Swift", year=2019,
    kms_driven=50_057, owner="Second Owner"))

print(f"Rs {v.price:,.0f}")          # Rs 4,91,831
print(f"{v.low:,.0f} - {v.high:,.0f}")
print(v.confidence, v.method)
print(v.notes)
```

If `models/` isn't in your current directory, pass an absolute path.

---

## 5. The notebook

```bash
cd carprice
pip install jupyter
jupyter notebook notebooks/car_valuation.ipynb
```

Then **Run All**. It downloads the dataset itself if missing, so `fetch`/`train` are not
prerequisites here.

Run it from `notebooks/` or the project root — the setup cell resolves paths from the
current directory and will fail if you launch it from somewhere unrelated.

---

## 6. Checking it installed correctly

```bash
python check_setup.py                                        # everything green?
python -c "import carprice; print(carprice.__version__)"     # 2.0.0
carvalue --help                                              # subcommands listed
cd carprice && pytest tests/ -q                              # 46 passed
```

`check_setup.py` is the one to remember — it reports the exact interpreter path, so if
`pip install` succeeded somewhere other than where you are running from, you will see it
immediately.

---

## 7. Stopping and restarting

**Stop the app:** Ctrl-C in the terminal running it.

**If Ctrl-C didn't work, or the port is stuck** (a crashed process still holding 5000):

```bash
# macOS / Linux
lsof -ti:5000 | xargs kill

# Windows PowerShell
netstat -ano | findstr :5000
taskkill /PID <pid_from_above> /F
```

**Restart after retraining.** The app loads the model once at startup. If you run
`carvalue train` while the app is up, the app keeps serving the old model until you
restart it. This is deliberate — reloading per request would add seconds to every
prediction.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| **`ModuleNotFoundError: No module named 'numpy'`** (or pandas / sklearn / joblib) | a bare `pip install -e .` did not complete, or you are in a different interpreter than the one you installed into | `python check_setup.py`, then `pip install -e ".[eda,web]"` with that same `python` |
| **`ModuleNotFoundError: No module named 'flask'`** | installed without the `web` extra | `pip install flask`, or `pip install -e ".[eda,web]"` |
| `ModuleNotFoundError: No module named 'carprice'` | not installed, or wrong venv | `pip install -e .` from `carprice/`, check `which python` |
| `No model at models/market_model.pkl` | haven't trained | `carvalue train` |
| `Raw dataset not found` | haven't fetched | `carvalue fetch` |
| `carvalue: command not found` | not on PATH | use `python -m carprice.carvalue ...` |
| `Address already in use` | port 5000 taken | `--port 5001`, or kill the old process above |
| `zsh: no matches found: .[eda,web]` | zsh globbing | quote it: `pip install -e ".[eda,web]"` |
| `python: command not found` | no `python` alias | use `python3` (or `py` on Windows) |
| Page loads but styling is broken | `static/` not found | launch with the full path to `web/app.py`, or `cd carprice` first |
| Valuation looks stale after retraining | app caches the model | restart the app |
| `Permission denied` on port 80/443 | ports under 1024 need root | use 5000+ |
| Windows Firewall prompt | `--host 0.0.0.0` exposes on LAN | allow it, or use `--host 127.0.0.1` |

---

## One-page cheat sheet

```bash
# one-time
cd carprice
pip install -e ".[eda,web]"
carvalue fetch
carvalue train

# web app
python web/app.py                  # -> http://127.0.0.1:5000
python web/app.py --port 5001      # if 5000 is busy

# one car
carvalue price --brand Kia --model Syros --year 2025 --km 6000

# notebook
jupyter notebook notebooks/car_valuation.ipynb

# tests
pytest tests/ -q
```

"""
catalogue.py
------------
A curated catalogue of the Indian car market: brand -> model -> (segment, price, specs).

Why this file exists
--------------------
A one-hot encoder over model names can only ever answer for the names it was trained on.
Ask it about a 2025 Kia Syros and it has no coefficient — it returns "an average car".
That is fine for a tutorial and useless for valuation.

This catalogue is the knowledge base that fixes it. Every model carries:

    segment         hatchback / sedan / suv / mpv / luxury-* / ev
    ex_showroom     current base ex-showroom price, in LAKH rupees (2026)
    engine_cc       typical engine displacement
    seats           typical seating

so an unseen car still gets valued from *what it is* (a 1.2L petrol compact SUV,
7 years old) instead of from a coefficient that never existed.

Prices are base/entry ex-showroom figures in lakh ₹, rounded, for the 2026 model year.
They are anchors, not quotes: the model blends them with real transaction data.
"""
from __future__ import annotations

from dataclasses import dataclass

#: segment -> short human label used in the UI
SEGMENT_LABELS = {
    "hatchback": "Hatchback",
    "sedan": "Sedan",
    "suv": "SUV",
    "mpv": "MPV / 7-seater",
    "luxury-hatch": "Luxury hatchback",
    "luxury-sedan": "Luxury sedan",
    "luxury-suv": "Luxury SUV",
    "ev-hatch": "Electric hatchback",
    "ev-sedan": "Electric sedan",
    "ev-suv": "Electric SUV",
    "pickup": "Pickup / utility",
}


@dataclass(frozen=True)
class ModelSpec:
    segment: str
    ex_showroom_lakh: float
    engine_cc: int
    seats: int


def _m(segment: str, price: float, cc: int, seats: int = 5) -> ModelSpec:
    return ModelSpec(segment, price, cc, seats)


# --------------------------------------------------------------------------- #
# The catalogue.
#
# Keyed by brand, then by the canonical model name. Order within a brand does not
# matter; matching is longest-name-first so "Grand i10" wins over "i10",
# "Innova Crysta" over "Innova", "Swift Dzire" over "Swift".
# --------------------------------------------------------------------------- #
CATALOGUE: dict[str, dict[str, ModelSpec]] = {
    "Maruti Suzuki": {
        "Alto 800":        _m("hatchback", 3.30, 796),
        "Alto K10":        _m("hatchback", 4.00, 998),
        "S-Presso":        _m("hatchback", 4.30, 998),
        "Celerio":         _m("hatchback", 5.40, 998),
        "Wagon R":         _m("hatchback", 5.80, 998),
        "Swift Dzire":     _m("sedan", 7.00, 1197),
        "Swift":           _m("hatchback", 6.60, 1197),
        "Dzire":           _m("sedan", 7.00, 1197),
        "Ignis":           _m("hatchback", 6.30, 1197),
        "Baleno":          _m("hatchback", 7.20, 1197),
        "Fronx":           _m("suv", 7.60, 1197),
        "Taisor":          _m("suv", 8.00, 1197),
        "Brezza":          _m("suv", 9.00, 1462),
        "Ertiga":          _m("mpv", 9.00, 1462, 7),
        "XL6":             _m("mpv", 12.20, 1462, 6),
        "Grand Vitara":    _m("suv", 11.50, 1462),
        "Jimny":           _m("suv", 12.70, 1462, 4),
        "Invicto":         _m("mpv", 25.20, 1987, 7),
        "eVitara":         _m("ev-suv", 17.50, 0),
        "Ciaz":            _m("sedan", 9.50, 1462),
        "Omni":            _m("mpv", 3.20, 796, 5),
        "Eeco":            _m("mpv", 5.60, 1197, 5),
        "Ritz":            _m("hatchback", 5.00, 1197),
        "SX4":             _m("sedan", 7.50, 1586),
    },
    "Hyundai": {
        "Grand i10 Nios":  _m("hatchback", 6.00, 1197),
        "Grand i10":       _m("hatchback", 6.00, 1197),
        "i10":             _m("hatchback", 5.50, 1086),
        "Santro":          _m("hatchback", 5.00, 1086),
        "Eon":             _m("hatchback", 4.20, 814),
        "Exter":           _m("suv", 6.20, 1086),
        "i20":             _m("hatchback", 7.50, 1197),
        "Aura":            _m("sedan", 6.50, 1197),
        "Xcent":           _m("sedan", 6.00, 1197),
        "Venue":           _m("suv", 8.00, 1197),
        "Verna":           _m("sedan", 11.00, 1497),
        "Creta":           _m("suv", 11.20, 1497),
        "Alcazar":         _m("suv", 14.99, 1497, 6),
        "Creta N Line":    _m("suv", 13.50, 1353),
        "Ioniq 5":         _m("ev-suv", 46.00, 0),
        "Kona Electric":   _m("ev-suv", 24.00, 0),
        "Elantra":         _m("sedan", 16.00, 1999),
        "Tucson":          _m("suv", 30.00, 1997),
    },
    "Tata": {
        "Tiago":           _m("hatchback", 5.70, 1199),
        "Tigor":           _m("sedan", 6.30, 1199),
        "Punch":           _m("suv", 6.20, 1199),
        "Altroz":          _m("hatchback", 7.00, 1199),
        "Nexon":           _m("suv", 8.00, 1199),
        "Curvv":           _m("suv", 10.00, 1199),
        "Sierra":          _m("suv", 11.50, 1199),
        "Harrier":         _m("suv", 15.50, 1956),
        "Safari":          _m("suv", 16.50, 1956, 6),
        "Indica":          _m("hatchback", 4.00, 1193),
        "Indica Vista":    _m("hatchback", 4.50, 1193),
        "Indigo":          _m("sedan", 5.00, 1193),
        "Indigo CS":       _m("sedan", 5.00, 1193),
        "Zest":            _m("sedan", 5.80, 1193),
        "Manza":           _m("sedan", 5.50, 1248),
        "Bolt":            _m("hatchback", 5.20, 1193),
        "Nano":            _m("hatchback", 2.60, 624),
        "Sumo":            _m("suv", 5.50, 1948, 7),
        "Xenon":           _m("pickup", 7.00, 2179),
        "Nexon EV":        _m("ev-suv", 14.50, 0),
        "Tiago EV":        _m("ev-hatch", 8.00, 0),
        "Tigor EV":        _m("ev-sedan", 8.50, 0),
        "Punch EV":        _m("ev-suv", 9.50, 0),
        "Curvv EV":        _m("ev-suv", 17.00, 0),
        "Harrier EV":      _m("ev-suv", 21.00, 0),
        "Safari EV":       _m("ev-suv", 22.00, 0, 6),
        "Avinya":          _m("ev-suv", 20.00, 0),
    },
    "Mahindra": {
        "Bolero":          _m("suv", 9.50, 2523, 7),
        "Bolero Neo":      _m("suv", 9.20, 2184, 7),
        "Bolero Power Plus": _m("suv", 8.50, 1493, 7),
        "KUV 100":         _m("suv", 5.50, 1198),
        "TUV 300":         _m("suv", 7.50, 1493, 7),
        "XUV 300":         _m("suv", 8.00, 1197),
        "XUV 3XO":         _m("suv", 8.00, 1197),
        "XUV 500":         _m("suv", 13.00, 2179, 7),
        "XUV 700":         _m("suv", 14.50, 1997, 7),
        "Scorpio":         _m("suv", 13.60, 2184, 7),
        "Scorpio N":       _m("suv", 13.60, 1997, 7),
        "Scorpio Classic": _m("suv", 13.60, 2184, 7),
        "Thar":            _m("suv", 11.50, 1997, 4),
        "Thar Roxx":       _m("suv", 12.99, 1997, 5),
        "Xylo":            _m("mpv", 8.50, 2489, 7),
        "Marazzo":         _m("mpv", 13.00, 1497, 7),
        "Quanto":          _m("suv", 6.00, 1493, 7),
        "TUV":             _m("suv", 7.50, 1493, 7),
        "Alturas G4":      _m("luxury-suv", 30.00, 2147, 7),
        "XEV 9e":          _m("ev-suv", 21.90, 0),
        "BE 6":            _m("ev-suv", 18.90, 0),
        "XUV400":          _m("ev-suv", 15.50, 0),
        "Verito":          _m("sedan", 6.00, 1461),
        "Logan":           _m("sedan", 5.50, 1461),
    },
    "Honda": {
        "Brio":            _m("hatchback", 5.00, 1198),
        "Jazz":            _m("hatchback", 7.50, 1199),
        "Amaze":           _m("sedan", 7.50, 1199),
        "City":            _m("sedan", 12.00, 1498),
        "City Hybrid":     _m("sedan", 19.00, 1498),
        "Verna":           _m("sedan", 11.00, 1497),
        "Civic":           _m("luxury-sedan", 22.00, 1799),
        "Accord":          _m("luxury-sedan", 40.00, 1993),
        "WR-V":            _m("suv", 9.00, 1199),
        "Elevate":         _m("suv", 11.70, 1498),
        "BR-V":            _m("mpv", 10.00, 1498, 7),
        "Mobilio":         _m("mpv", 7.50, 1498, 7),
        "CR-V":            _m("luxury-suv", 30.00, 1997),
        "HR-V":            _m("suv", 15.00, 1498),
    },
    "Toyota": {
        "Etios":           _m("sedan", 6.50, 1364),
        "Etios Liva":      _m("hatchback", 6.00, 1197),
        "Etios Cross":     _m("hatchback", 6.50, 1364),
        "Liva":            _m("hatchback", 6.00, 1197),
        "Glanza":          _m("hatchback", 7.00, 1197),
        "Taisor":          _m("suv", 8.00, 1197),
        "Urban Cruiser":   _m("suv", 9.00, 1462),
        "Urban Cruiser Hyryder": _m("suv", 11.50, 1462),
        "Rumion":          _m("mpv", 10.50, 1462, 7),
        "Innova":          _m("mpv", 19.00, 2393, 7),
        "Innova Crysta":   _m("mpv", 19.50, 2393, 7),
        "Innova Hycross":  _m("mpv", 19.00, 1987, 7),
        "Corolla":         _m("sedan", 17.00, 1598),
        "Corolla Altis":   _m("sedan", 17.00, 1598),
        "Yaris":           _m("sedan", 9.50, 1496),
        "Camry":           _m("luxury-sedan", 48.00, 2487),
        "Fortuner":        _m("suv", 33.00, 2755, 7),
        "Fortuner Legender": _m("suv", 36.00, 2755, 7),
        "Qualis":          _m("mpv", 7.00, 2446, 7),
        "Hilux":           _m("pickup", 30.00, 2755),
        "Vellfire":        _m("luxury-suv", 95.00, 2487, 7),
        "Land Cruiser":    _m("luxury-suv", 210.00, 3346, 7),
    },
    "Kia": {
        "Sonet":           _m("suv", 8.00, 1197),
        "Syros":           _m("suv", 9.00, 1197),
        "Seltos":          _m("suv", 11.00, 1497),
        "Carens":          _m("mpv", 11.00, 1497, 7),
        "Carnival":        _m("mpv", 32.00, 2199, 7),
        "EV6":             _m("ev-suv", 66.00, 0),
        "EV9":             _m("ev-suv", 130.00, 0, 7),
        "Carens Clavis":   _m("mpv", 11.50, 1497, 7),
    },
    "Renault": {
        "Kwid":            _m("hatchback", 4.70, 999),
        "Triber":          _m("mpv", 6.00, 999, 7),
        "Kiger":           _m("suv", 6.00, 999),
        "Duster":          _m("suv", 9.00, 1498),
        "Lodgy":           _m("mpv", 8.50, 1461, 7),
        "Pulse":           _m("hatchback", 5.00, 1198),
        "Scala":           _m("sedan", 7.00, 1498),
        "Captur":          _m("suv", 11.00, 1498),
        "Fluence":         _m("sedan", 14.00, 1598),
    },
    "Nissan": {
        "Micra":           _m("hatchback", 6.00, 1198),
        "Sunny":           _m("sedan", 7.00, 1498),
        "Terrano":         _m("suv", 10.00, 1461),
        "Kicks":           _m("suv", 10.00, 1498),
        "Magnite":         _m("suv", 6.00, 999),
        "Tekna":           _m("suv", 7.00, 999),
        "X-Trail":         _m("luxury-suv", 50.00, 2498),
        "GTR":             _m("luxury-suv", 220.00, 3799),
    },
    "Volkswagen": {
        "Polo":            _m("hatchback", 8.00, 999),
        "Ameo":            _m("sedan", 6.50, 1197),
        "Vento":           _m("sedan", 10.00, 999),
        "Virtus":          _m("sedan", 11.50, 999),
        "Taigun":          _m("suv", 11.50, 999),
        "Tiguan":          _m("luxury-suv", 35.00, 1984),
        "T-Roc":           _m("suv", 21.00, 1498),
        "Jetta":           _m("luxury-sedan", 20.00, 1395),
        "Passat":          _m("luxury-sedan", 30.00, 1984),
        "Touareg":         _m("luxury-suv", 75.00, 2967),
        "Touran":          _m("mpv", 20.00, 1598, 7),
    },
    "Skoda": {
        "Fabia":           _m("hatchback", 6.50, 1197),
        "Rapid":           _m("sedan", 9.00, 999),
        "Slavia":          _m("sedan", 11.00, 999),
        "Laura":           _m("luxury-sedan", 17.00, 1896),
        "Octavia":         _m("luxury-sedan", 28.00, 1984),
        "Superb":          _m("luxury-sedan", 40.00, 1984),
        "Kushaq":          _m("suv", 11.00, 999),
        "Kylaq":           _m("suv", 8.00, 999),
        "Karoq":           _m("luxury-suv", 28.00, 1498),
        "Kodiaq":          _m("luxury-suv", 40.00, 1984, 7),
        "Yeti":            _m("suv", 18.00, 1968),
    },
    "Ford": {
        "Figo":            _m("hatchback", 6.00, 1194),
        "Figo Aspire":     _m("sedan", 7.00, 1194),
        "Freestyle":       _m("hatchback", 7.00, 1194),
        "Fiesta":          _m("sedan", 8.50, 1499),
        "Fiesta Classic":  _m("sedan", 7.00, 1399),
        "Ikon":            _m("sedan", 5.50, 1396),
        "EcoSport":        _m("suv", 9.00, 1496),
        "Endeavour":       _m("luxury-suv", 33.00, 2198, 7),
        "Mustang":         _m("luxury-suv", 80.00, 5038, 4),
    },
    "Chevrolet": {
        "Spark":           _m("hatchback", 4.00, 995),
        "Beat":            _m("hatchback", 5.00, 1199),
        "Sail":            _m("sedan", 6.00, 1199),
        "Sail Hatchback":  _m("hatchback", 5.50, 1199),
        "Aveo":            _m("sedan", 6.00, 1150),
        "Cruze":           _m("luxury-sedan", 15.00, 1998),
        "Optra":           _m("sedan", 8.00, 1598),
        "Tavera":          _m("mpv", 7.00, 2499, 7),
        "Enjoy":           _m("mpv", 6.50, 1248, 7),
        "Captiva":         _m("luxury-suv", 22.00, 2231, 7),
        "Trailblazer":     _m("luxury-suv", 27.00, 2776, 7),
    },
    "Fiat": {
        "Punto":           _m("hatchback", 5.50, 1172),
        "Punto Evo":       _m("hatchback", 6.00, 1248),
        "Linea":           _m("sedan", 7.00, 1248),
        "Palio":           _m("hatchback", 4.50, 1172),
        "Avventura":       _m("hatchback", 6.50, 1248),
        "Urban Cross":     _m("hatchback", 7.00, 1248),
    },
    "Datsun": {
        "GO":              _m("hatchback", 4.00, 1198),
        "GO Plus":         _m("mpv", 4.50, 1198, 7),
        "Redi-GO":         _m("hatchback", 3.00, 799),
    },
    "MG": {
        "Hector":          _m("suv", 15.00, 1451),
        "Hector Plus":     _m("suv", 17.50, 1451, 6),
        "Astor":           _m("suv", 10.00, 1498),
        "ZS EV":           _m("ev-suv", 18.50, 0),
        "Comet EV":        _m("ev-hatch", 7.00, 0, 4),
        "Windsor EV":      _m("ev-suv", 14.00, 0),
        "Gloster":         _m("luxury-suv", 40.00, 1996, 7),
        "Cyberster":       _m("luxury-suv", 65.00, 0, 2),
        "MGS5":            _m("ev-suv", 25.00, 0),
    },
    "Jeep": {
        "Compass":         _m("suv", 20.00, 1956),
        "Meridian":        _m("suv", 25.00, 1956, 7),
        "Wrangler":        _m("luxury-suv", 70.00, 1995, 4),
        "Grand Cherokee":  _m("luxury-suv", 80.00, 2987, 5),
    },
    "BMW": {
        "1 Series":        _m("luxury-hatch", 45.00, 1998),
        "2 Series":        _m("luxury-sedan", 45.00, 1998),
        "3 Series":        _m("luxury-sedan", 60.00, 1998),
        "3 Series Gran Limousine": _m("luxury-sedan", 63.00, 1998),
        "5 Series":        _m("luxury-sedan", 75.00, 1998),
        "6 Series":        _m("luxury-sedan", 75.00, 2998),
        "7 Series":        _m("luxury-sedan", 180.00, 2998),
        "X1":              _m("luxury-suv", 50.00, 1499),
        "X2":              _m("luxury-suv", 55.00, 1998),
        "X3":              _m("luxury-suv", 75.00, 1998),
        "X4":              _m("luxury-suv", 80.00, 1998),
        "X5":              _m("luxury-suv", 100.00, 2998),
        "X6":              _m("luxury-suv", 110.00, 2998),
        "X7":              _m("luxury-suv", 130.00, 2998, 7),
        "i3":              _m("ev-sedan", 60.00, 0),
        "i4":              _m("ev-sedan", 75.00, 0),
        "iX":              _m("ev-suv", 130.00, 0),
        "Z4":              _m("luxury-sedan", 90.00, 1998, 2),
    },
    "Mercedes-Benz": {
        "A-Class":         _m("luxury-hatch", 45.00, 1332),
        "B-Class":         _m("luxury-hatch", 50.00, 1595),
        "C-Class":         _m("luxury-sedan", 60.00, 1496),
        "E-Class":         _m("luxury-sedan", 85.00, 1993),
        "S-Class":         _m("luxury-sedan", 180.00, 2999),
        "CLA":             _m("luxury-sedan", 50.00, 1332),
        "GLA":             _m("luxury-suv", 55.00, 1332),
        "GLB":             _m("luxury-suv", 65.00, 1332, 7),
        "GLC":             _m("luxury-suv", 80.00, 1993),
        "GLE":             _m("luxury-suv", 100.00, 2999),
        "GLS":             _m("luxury-suv", 130.00, 2999, 7),
        "G-Class":         _m("luxury-suv", 200.00, 2925),
        "M-Class":         _m("luxury-suv", 60.00, 2987),
        "EQS":             _m("ev-sedan", 160.00, 0),
        "EQE":             _m("ev-sedan", 130.00, 0),
        "EQB":             _m("ev-suv", 80.00, 0),
        "EQA":             _m("ev-suv", 66.00, 0),
        "V-Class":         _m("luxury-suv", 100.00, 1950, 7),
    },
    "Audi": {
        "A3":              _m("luxury-sedan", 45.00, 1395),
        "A4":              _m("luxury-sedan", 55.00, 1984),
        "A6":              _m("luxury-sedan", 70.00, 1984),
        "A8":              _m("luxury-sedan", 150.00, 2995),
        "Q3":              _m("luxury-suv", 50.00, 1395),
        "Q5":              _m("luxury-suv", 70.00, 1984),
        "Q7":              _m("luxury-suv", 90.00, 2967, 7),
        "Q8":              _m("luxury-suv", 120.00, 2995),
        "e-tron":          _m("ev-suv", 120.00, 0),
        "e-tron GT":       _m("ev-sedan", 180.00, 0),
        "TT":              _m("luxury-sedan", 70.00, 1984, 4),
    },
    "Jaguar": {
        "XE":              _m("luxury-sedan", 45.00, 1999),
        "XF":              _m("luxury-sedan", 60.00, 1999),
        "XJ":              _m("luxury-sedan", 100.00, 2993),
        "F-Pace":          _m("luxury-suv", 75.00, 1997),
        "E-Pace":          _m("luxury-suv", 60.00, 1997),
        "I-Pace":          _m("ev-suv", 110.00, 0),
        "F-Type":          _m("luxury-sedan", 110.00, 2995, 2),
    },
    "Volvo": {
        "S60":             _m("luxury-sedan", 50.00, 1969),
        "S90":             _m("luxury-sedan", 70.00, 1969),
        "V40":             _m("luxury-hatch", 30.00, 1969),
        "V60":             _m("luxury-sedan", 55.00, 1969),
        "XC40":            _m("luxury-suv", 50.00, 1969),
        "XC60":            _m("luxury-suv", 70.00, 1969),
        "XC90":            _m("luxury-suv", 100.00, 1969, 7),
        "C40":             _m("ev-suv", 60.00, 0),
        "EX30":            _m("ev-suv", 45.00, 0),
        "EX40":            _m("ev-suv", 55.00, 0),
    },
    "Land Rover": {
        "Defender":        _m("luxury-suv", 100.00, 1997, 5),
        "Discovery":       _m("luxury-suv", 80.00, 1997, 7),
        "Discovery Sport": _m("luxury-suv", 70.00, 1997, 7),
        "Range Rover Evoque": _m("luxury-suv", 70.00, 1997),
        "Range Rover Velar": _m("luxury-suv", 90.00, 1997),
        "Range Rover Sport": _m("luxury-suv", 140.00, 2996),
        "Range Rover":     _m("luxury-suv", 250.00, 2996),
    },
    "Lexus": {
        "ES":              _m("luxury-sedan", 65.00, 2487),
        "LS":              _m("luxury-sedan", 180.00, 3456),
        "NX":              _m("luxury-suv", 70.00, 2487),
        "RX":              _m("luxury-suv", 95.00, 2393),
        "LX":              _m("luxury-suv", 250.00, 3445, 7),
        "UX":              _m("luxury-suv", 45.00, 1987),
    },
    "MINI": {
        "Cooper":          _m("luxury-hatch", 45.00, 1499),
        "Countryman":      _m("luxury-suv", 50.00, 1499),
        "Clubman":         _m("luxury-hatch", 45.00, 1499),
    },
    "Porsche": {
        "Macan":           _m("luxury-suv", 90.00, 1984),
        "Cayenne":         _m("luxury-suv", 140.00, 2995),
        "911":             _m("luxury-sedan", 200.00, 2981, 2),
        "Panamera":        _m("luxury-sedan", 180.00, 2894),
        "Taycan":          _m("ev-sedan", 160.00, 0),
        "718":             _m("luxury-sedan", 100.00, 1988, 2),
    },
    "Citroen": {
        "C3":              _m("hatchback", 6.50, 1199),
        "eC3":             _m("ev-hatch", 12.00, 0),
        "C3 Aircross":     _m("suv", 11.00, 1199, 7),
        "C5 Aircross":     _m("luxury-suv", 35.00, 1499),
        "Basalt":          _m("suv", 8.00, 1199),
    },
    "BYD": {
        "Atto 3":          _m("ev-suv", 25.00, 0),
        "e6":              _m("ev-suv", 30.00, 0),
        "Seal":            _m("ev-sedan", 41.00, 0),
        "Sealion 7":       _m("ev-suv", 48.00, 0),
        "M9":              _m("ev-suv", 90.00, 0, 7),
    },
    "Isuzu": {
        "D-Max":           _m("pickup", 20.00, 1898),
        "MU-X":            _m("luxury-suv", 35.00, 1898, 7),
        "S-Cab":           _m("pickup", 15.00, 2499),
    },
    "Force Motors": {
        "Gurkha":          _m("suv", 16.00, 2596),
        "Trax":            _m("suv", 8.00, 2596, 7),
    },
    "Mitsubishi": {
        "Pajero":          _m("luxury-suv", 30.00, 2835, 7),
        "Lancer":          _m("sedan", 9.00, 1468),
        "Outlander":       _m("luxury-suv", 25.00, 2360, 7),
    },
    "Hindustan Motors": {
        "Ambassador":      _m("sedan", 5.00, 1489),
        "Contessa":        _m("sedan", 5.00, 1489),
    },
    "Daewoo": {
        "Matiz":           _m("hatchback", 3.00, 796),
        "Nexia":           _m("sedan", 4.50, 1498),
    },
    "Opel": {
        "Astra":           _m("sedan", 6.00, 1598),
        "Corsa":           _m("hatchback", 5.00, 1389),
        "Sail":            _m("sedan", 5.50, 1398),
    },
    "Peugeot": {
        "309":             _m("sedan", 4.00, 1580),
        "206":             _m("hatchback", 4.50, 1360),
    },
    "Ashok Leyland": {
        "Stile":           _m("mpv", 5.50, 1461, 7),
        "Dost":            _m("pickup", 6.00, 1478),
    },
    "Premier": {
        "Rl":              _m("hatchback", 4.50, 1173),
        "Rio":             _m("hatchback", 5.00, 1173),
    },
}

#: Brand aliases seen in listings -> catalogue key
BRAND_ALIASES = {
    "maruti": "Maruti Suzuki",
    "maruti suzuki": "Maruti Suzuki",
    "marutisuzuki": "Maruti Suzuki",
    "hyundai": "Hyundai",
    "tata": "Tata",
    "tatamotors": "Tata",
    "mahindra": "Mahindra",
    "honda": "Honda",
    "toyota": "Toyota",
    "kia": "Kia",
    "renault": "Renault",
    "nissan": "Nissan",
    "volkswagen": "Volkswagen",
    "vw": "Volkswagen",
    "skoda": "Skoda",
    "ford": "Ford",
    "chevrolet": "Chevrolet",
    "chevy": "Chevrolet",
    "fiat": "Fiat",
    "datsun": "Datsun",
    "mg": "MG",
    "jeep": "Jeep",
    "bmw": "BMW",
    "mercedes": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "mercedesbenz": "Mercedes-Benz",
    "audi": "Audi",
    "jaguar": "Jaguar",
    "volvo": "Volvo",
    "land rover": "Land Rover",
    "landrover": "Land Rover",
    "lexus": "Lexus",
    "mini": "MINI",
    "porsche": "Porsche",
    "citroen": "Citroen",
    "byd": "BYD",
    "isuzu": "Isuzu",
    "force": "Force Motors",
    "force motors": "Force Motors",
    "mitsubishi": "Mitsubishi",
    "ambassador": "Hindustan Motors",
    "hindustan": "Hindustan Motors",
    "daewoo": "Daewoo",
    "opel": "Opel",
    "peugeot": "Peugeot",
    "ashok leyland": "Ashok Leyland",
    "premier": "Premier",
    "tesla": "Tesla",
    "tesla motors": "Tesla",
}

#: Brands we know of but have no catalogue yet — valued on segment + age alone
UNCATALOGUED_BRANDS = {
    # Tesla entered India in 2026 with imported CBUs; Model Y RWD lands ~Rs 55 lakh
    "Tesla": {"Model 3": _m("ev-sedan", 55.00, 0),
              "Model Y": _m("ev-suv", 56.00, 0),
              "Model S": _m("ev-sedan", 110.00, 0),
              "Model X": _m("ev-suv", 120.00, 0)},
}

CATALOGUE.update(UNCATALOGUED_BRANDS)


def resolve_brand(raw: str) -> str | None:
    """Map a raw brand string ('maruti', 'Mercedes-Benz', 'MARUTI SUZUKI') to a key."""
    if raw is None:
        return None
    text = " ".join(str(raw).strip().lower().split())
    if text in BRAND_ALIASES:
        return BRAND_ALIASES[text]
    # first token fallback: 'Maruti Suzuki Swift' -> 'maruti' -> Maruti Suzuki
    first = text.split()[0] if text.split() else ""
    return BRAND_ALIASES.get(first)


def _name_pattern(option: str) -> "re.Pattern[str]":
    """
    Word-boundary pattern for a catalogue name, tolerant of the punctuation listings use.

    Matching on boundaries is not optional: a naive substring test maps 'Kia Clavis'
    onto the Mercedes 'CLA' (because "cla" sits inside "clavis") and values a ₹9 lakh
    hatchback at ₹42 lakh.
    """
    import re
    tokens = re.split(r"[\s\-]+", option.lower().strip())
    # each separator between tokens may appear as a space, a hyphen, or nothing at all,
    # so 'XUV 500' matches 'XUV500', 'Redi-Go' matches 'Redi Go', 'S-Presso' matches 'S Presso'
    body = r"[\s\-]*".join(re.escape(t) for t in tokens if t)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


_PATTERN_CACHE: dict[str, "re.Pattern[str]"] = {}


def _longest_match(text: str, options: list[str], *, allow_lead: bool = False) -> str | None:
    """
    Longest catalogue name that occurs in `text` on word boundaries.

    `allow_lead` additionally accepts a match on the first token, which is how a bare
    model entry should behave: the user typing 'Clavis' means Kia's new SUV, not
    'Carens Clavis' (a different car), and 'Creta' should find 'Creta N Line'.
    """
    low = " ".join(str(text).lower().split())
    hits: list[tuple[int, str]] = []

    def _pattern(option: str):
        pat = _PATTERN_CACHE.get(option)
        if pat is None:
            pat = _name_pattern(option)
            _PATTERN_CACHE[option] = pat
        return pat

    def _lead_pattern(option: str):
        key = "\x00" + option
        pat = _PATTERN_CACHE.get(key)
        if pat is None:
            import re
            first = re.split(r"[\s\-]+", option.lower().strip())[0]
            pat = re.compile(rf"(?<![a-z0-9]){re.escape(first)}(?![a-z0-9])")
            _PATTERN_CACHE[key] = pat
        return pat

    for option in options:
        m = _pattern(option).search(low)
        if m:
            hits.append((len(m.group(0)), option))
            continue
        if allow_lead:
            m = _lead_pattern(option).search(low)
            if m:
                # a lead-token match is weaker than a full-name match
                hits.append((len(m.group(0)) - 1, option))
    if not hits:
        return None
    hits.sort(reverse=True)
    return hits[0][1]


def resolve_model(brand: str | None, raw_name: str) -> tuple[str, ModelSpec] | None:
    """
    Find (canonical_model, spec) for a listing name, or None.

    Tries the given brand's catalogue first, then every brand (so a stray
    'Hyundai Creta' tagged under the wrong brand still resolves).
    """
    if raw_name is None:
        return None
    text = str(raw_name)

    if brand and brand in CATALOGUE:
        hit = _longest_match(text, list(CATALOGUE[brand]), allow_lead=True)
        if hit:
            return hit, CATALOGUE[brand][hit]

    for other, models in CATALOGUE.items():
        if other == brand:
            continue
        hit = _longest_match(text, list(models), allow_lead=True)
        if hit:
            return hit, models[hit]

    return None


def brand_of(raw_name: str) -> str | None:
    """Pull the brand out of a full listing name like 'Maruti Swift VDI'."""
    if not raw_name:
        return None
    words = str(raw_name).replace("-", " ").split()
    for n in (3, 2, 1):                      # 'Land Rover', 'Maruti Suzuki', 'Maruti'
        cand = " ".join(words[:n]).lower()
        if cand in BRAND_ALIASES:
            return BRAND_ALIASES[cand]
    return BRAND_ALIASES.get(words[0].lower()) if words else None


def all_brands() -> list[str]:
    return sorted(CATALOGUE)


def models_for(brand: str) -> list[str]:
    return sorted(CATALOGUE.get(brand, {}))

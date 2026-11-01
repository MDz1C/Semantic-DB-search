"""Deterministic generator of a realistic European fund and ETF universe.

Feeds the bundled "Try with sample data" workbook, the tests and ``tools/evaluate.py``.
The Brokerage column holds the management *company* while fund names carry a *brand*
that may differ from it (BlackRock -> iShares, DWS -> Xtrackers, Amundi -> Lyxor), be
abbreviated (ISHS, Xtr.), uppercased or absent: the identity confusion the app must
resolve.  The same exposure (S&P 500, MSCI World...) exists at many brokerages, and
the universe scales to tens of thousands of unique rows through series, share classes,
hedges and currencies, as a real fund database does.

    .venv/Scripts/python.exe tools/make_sample_data.py --rows 2500 --out assets/sample/sample_funds.xlsx
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import heapq
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semdb.core.text import normalize, tokens  # noqa: E402

DEFAULT_OUT = ROOT / "assets" / "sample" / "sample_funds.xlsx"
COLUMNS = [
    "ISIN", "Ticker", "Fund name", "Brokerage", "Category", "Asset class", "Region",
    "Currency", "Distribution", "Domicile", "TER (%)", "AUM (M EUR)", "Launch date",
]
NUMBER_COLUMNS = {"TER (%)": "0.00", "AUM (M EUR)": "#,##0.0"}
DATE_COLUMNS = {"Launch date"}

# --- categories (Spanish, as in a Spanish distributor's database) ----------------------
RVI = "Renta Variable Internacional"
RVI_EU = "Renta Variable Internacional Europa"
RVI_US = "Renta Variable Internacional EE.UU."
RVI_JP = "Renta Variable Internacional Japón"
RVI_EM = "Renta Variable Internacional Emergentes"
RV_EURO = "Renta Variable Euro"
RV_NAC = "Renta Variable Nacional"
RV_SECT = "Renta Variable Sectorial"
RF_EURO = "Renta Fija Euro"
RF_CP = "Renta Fija Euro Corto Plazo"
RF_INT = "Renta Fija Internacional"
MONET = "Monetario"
MIX_C = "Mixto Conservador"
MIX_M = "Mixto Moderado"
MIX_A = "Mixto Agresivo"
MAT = "Materias Primas"
RENT_OBJ = "Rentabilidad Objetivo"

# category -> (asset class, default region, typical TER % of a retail mutual fund class)
CATEGORIES = {
    RVI: ("Equity", "Global", 1.7), RVI_EU: ("Equity", "Europe", 1.6),
    RVI_US: ("Equity", "United States", 1.5), RVI_JP: ("Equity", "Japan", 1.4),
    RVI_EM: ("Equity", "Emerging Markets", 1.8), RV_EURO: ("Equity", "Eurozone", 1.5),
    RV_NAC: ("Equity", "Spain", 1.6), RV_SECT: ("Equity", "Global", 1.8),
    RF_EURO: ("Fixed Income", "Eurozone", 0.75), RF_CP: ("Fixed Income", "Eurozone", 0.4),
    RF_INT: ("Fixed Income", "Global", 0.9), MONET: ("Money Market", "Eurozone", 0.25),
    MIX_C: ("Multi-Asset", "Global", 1.1), MIX_M: ("Multi-Asset", "Global", 1.35),
    MIX_A: ("Multi-Asset", "Global", 1.6), MAT: ("Commodity", "Global", 0.6),
    RENT_OBJ: ("Fixed Income", "Eurozone", 0.45),
}
DOMICILES = {
    "IE": "Ireland", "LU": "Luxembourg", "DE": "Germany", "FR": "France",
    "ES": "Spain", "NL": "Netherlands", "JE": "Jersey",
}


@dataclass(frozen=True, slots=True)
class Exposure:
    """An index or theme a product tracks."""

    name: str
    region: str
    category: str
    currency: str
    ter: float  # typical TER of a plain ETF on it
    popularity: float
    group: str = "plain"  # which modifiers make sense: broad / esg / country / bond / plain


X = Exposure  # keeps the table below readable
COMMON = (
    X("S&P 500", "United States", RVI_US, "USD", 0.07, 10, "broad"),
    X("MSCI World", "Global", RVI, "USD", 0.20, 10, "broad"),
    X("MSCI ACWI", "Global", RVI, "USD", 0.20, 6, "broad"),
    X("MSCI Emerging Markets", "Emerging Markets", RVI_EM, "USD", 0.18, 8, "broad"),
    X("MSCI EM IMI", "Emerging Markets", RVI_EM, "USD", 0.18, 4, "esg"),
    X("FTSE All-World", "Global", RVI, "USD", 0.22, 6, "esg"),
    X("FTSE Developed World", "Global", RVI, "USD", 0.12, 3, "esg"),
    X("FTSE Emerging Markets", "Emerging Markets", RVI_EM, "USD", 0.22, 3),
    X("Euro Stoxx 50", "Eurozone", RV_EURO, "EUR", 0.10, 7, "broad"),
    X("STOXX Europe 600", "Europe", RVI_EU, "EUR", 0.20, 6, "broad"),
    X("MSCI Europe", "Europe", RVI_EU, "EUR", 0.12, 6, "broad"),
    X("MSCI EMU", "Eurozone", RV_EURO, "EUR", 0.12, 4, "broad"),
    X("MSCI USA", "United States", RVI_US, "USD", 0.07, 5, "broad"),
    X("Nasdaq 100", "United States", RVI_US, "USD", 0.30, 7, "esg"),
    X("Russell 2000", "United States", RVI_US, "USD", 0.30, 3),
    X("MSCI Japan", "Japan", RVI_JP, "USD", 0.12, 5, "broad"),
    X("MSCI China", "China", RVI_EM, "USD", 0.28, 4, "esg"),
    X("MSCI India", "India", RVI_EM, "USD", 0.65, 4),
    X("MSCI Pacific ex Japan", "Asia Pacific", RVI, "USD", 0.20, 2),
    X("MSCI Korea", "South Korea", RVI_EM, "USD", 0.45, 2),
    X("MSCI Brazil", "Brazil", RVI_EM, "USD", 0.65, 2),
    X("FTSE 100", "United Kingdom", RVI_EU, "GBP", 0.07, 3, "country"),
    X("DAX", "Germany", RV_EURO, "EUR", 0.09, 4, "country"),
    X("IBEX 35", "Spain", RV_NAC, "EUR", 0.30, 4, "country"),
    X("CAC 40", "France", RV_EURO, "EUR", 0.25, 3, "country"),
    X("MSCI World Small Cap", "Global", RVI, "USD", 0.35, 4, "esg"),
    X("Global Clean Energy", "Global", RV_SECT, "USD", 0.65, 4),
    X("MSCI World Information Technology", "Global", RV_SECT, "USD", 0.25, 3),
    X("MSCI World Health Care", "Global", RV_SECT, "USD", 0.25, 3),
    X("S&P Global Dividend Aristocrats", "Global", RVI, "USD", 0.45, 2),
    X("FTSE EPRA Nareit Developed", "Global", RV_SECT, "USD", 0.40, 2),
    X("Global Aggregate Bond", "Global", RF_INT, "USD", 0.10, 5, "bond"),
    X("EUR Govt Bond", "Eurozone", RF_EURO, "EUR", 0.09, 4, "bond"),
    X("EUR Govt Bond 1-3yr", "Eurozone", RF_CP, "EUR", 0.15, 5, "bond"),
    X("EUR Govt Bond 7-10yr", "Eurozone", RF_EURO, "EUR", 0.15, 3, "bond"),
    X("USD Treasury Bond", "United States", RF_INT, "USD", 0.07, 3, "bond"),
    X("USD Treasury Bond 1-3yr", "United States", RF_INT, "USD", 0.07, 4, "bond"),
    X("USD Treasury Bond 7-10yr", "United States", RF_INT, "USD", 0.07, 3, "bond"),
    X("USD Treasury Bond 20+yr", "United States", RF_INT, "USD", 0.07, 3, "bond"),
    X("EUR Corporate Bond", "Eurozone", RF_EURO, "EUR", 0.20, 5, "bond"),
    X("USD Corporate Bond", "United States", RF_INT, "USD", 0.20, 3, "bond"),
    X("USD High Yield Corporate Bond", "United States", RF_INT, "USD", 0.50, 3, "bond"),
    X("EUR High Yield Corporate Bond", "Eurozone", RF_EURO, "EUR", 0.50, 3, "bond"),
    X("Euro Inflation Linked Govt Bond", "Eurozone", RF_EURO, "EUR", 0.09, 2, "bond"),
    X("USD Emerging Markets Bond", "Emerging Markets", RF_INT, "USD", 0.45, 3, "bond"),
    X("EUR Ultrashort Bond", "Eurozone", RF_CP, "EUR", 0.09, 2),
    X("EUR Overnight Rate", "Eurozone", MONET, "EUR", 0.10, 3),
    X("Physical Gold", "Global", MAT, "USD", 0.12, 5),
    X("Physical Silver", "Global", MAT, "USD", 0.20, 2),
    X("Bloomberg Commodity", "Global", MAT, "USD", 0.25, 2),
)
COMMON_BY_NAME = {exposure.name: exposure for exposure in COMMON}

# modifier -> (position in the name, popularity factor, TER factor)
MODIFIERS = {
    "Core": ("prefix", 1.3, 0.6),
    "ESG": ("suffix", 0.5, 1.15),
    "SRI": ("suffix", 0.35, 1.2),
    "ESG Screened": ("suffix", 0.3, 1.0),
    "Paris-Aligned Climate": ("suffix", 0.25, 1.1),
    "Equal Weight": ("suffix", 0.25, 1.3),
    "Quality Factor": ("suffix", 0.2, 1.25),
    "Value Factor": ("suffix", 0.2, 1.25),
    "Momentum Factor": ("suffix", 0.2, 1.25),
    "Min Vol": ("suffix", 0.2, 1.25),
}
GROUP_MODIFIERS = {
    "broad": tuple(MODIFIERS),
    "esg": ("ESG", "SRI", "ESG Screened", "Paris-Aligned Climate"),
    "country": ("Core", "ESG"),
    "bond": ("Core", "ESG", "Paris-Aligned Climate"),
    "plain": (),
}


@dataclass(frozen=True, slots=True)
class EtfLine:
    """An ETF range: one brand of one company."""

    company: str
    brand: str
    style: str  # share-class naming style (CLASS_LABELS key)
    domiciles: tuple[str, ...]
    weight: float
    breadth: float  # how much of COMMON it covers
    modifiers: tuple[str, ...] = ()
    own: tuple[Exposure, ...] = ()  # brand-specific products; "{year}" expands to a maturity series
    covers: tuple[str, ...] = ()  # explicit COMMON coverage instead of ``breadth``
    currency: str = ""  # base currency override
    ter_factor: float = 1.0


def _own(*items: tuple) -> tuple[Exposure, ...]:
    return tuple(Exposure(*item) for item in items)


ETF_LINES = (
    EtfLine("BlackRock", "iShares", "ishares", ("IE",), 10, 1.15, tuple(MODIFIERS), _own(
        ("Automation & Robotics", "Global", RV_SECT, "USD", 0.40, 3),
        ("Healthcare Innovation", "Global", RV_SECT, "USD", 0.40, 2),
        ("Digital Security", "Global", RV_SECT, "USD", 0.40, 2),
        ("Global Water", "Global", RV_SECT, "USD", 0.65, 2),
        ("Global Infrastructure", "Global", RV_SECT, "USD", 0.65, 2),
        ("Electric Vehicles and Driving Technology", "Global", RV_SECT, "USD", 0.40, 1),
        ("iBonds Dec {year} Term EUR Corp", "Eurozone", RF_EURO, "EUR", 0.12, 6),
        ("iBonds Dec {year} Term USD Corp", "United States", RF_INT, "USD", 0.10, 4))),
    EtfLine("Vanguard", "Vanguard", "vanguard", ("IE",), 7, 1.0, ("ESG",), _own(
        ("LifeStrategy 20% Equity", "Global", MIX_C, "EUR", 0.25, 2),
        ("LifeStrategy 40% Equity", "Global", MIX_C, "EUR", 0.25, 2),
        ("LifeStrategy 60% Equity", "Global", MIX_M, "EUR", 0.25, 3),
        ("LifeStrategy 80% Equity", "Global", MIX_A, "EUR", 0.25, 3),
        ("ESG Global All Cap", "Global", RVI, "USD", 0.24, 3),
        ("FTSE Developed Europe", "Europe", RVI_EU, "EUR", 0.10, 3),
        ("FTSE North America", "United States", RVI_US, "USD", 0.10, 2),
        ("FTSE Japan", "Japan", RVI_JP, "USD", 0.15, 2),
        ("FTSE 250", "United Kingdom", RVI_EU, "GBP", 0.10, 1),
        ("U.K. Gilt", "United Kingdom", RF_INT, "GBP", 0.07, 1)),
        covers=("S&P 500", "FTSE All-World", "FTSE Developed World", "FTSE Emerging Markets",
                "FTSE 100", "Global Aggregate Bond", "EUR Govt Bond", "USD Treasury Bond",
                "EUR Corporate Bond", "USD Corporate Bond", "USD Emerging Markets Bond"),
        ter_factor=0.8),
    EtfLine("Amundi", "Amundi", "amundi", ("LU", "IE", "FR"), 6, 1.0,
            ("ESG", "SRI", "Paris-Aligned Climate", "Equal Weight"), _own(
        ("Prime Global", "Global", RVI, "EUR", 0.05, 3),
        ("Prime USA", "United States", RVI_US, "EUR", 0.05, 2),
        ("Prime Europe", "Europe", RVI_EU, "EUR", 0.05, 2),
        ("Prime Japan", "Japan", RVI_JP, "EUR", 0.05, 1),
        ("Prime Emerging Markets", "Emerging Markets", RVI_EM, "EUR", 0.10, 2),
        ("Prime Euro Govies", "Eurozone", RF_EURO, "EUR", 0.05, 2)), currency="EUR", ter_factor=0.9),
    EtfLine("Amundi", "Lyxor", "lyxor", ("FR", "LU"), 5, 0.95, ("Core", "ESG", "Paris-Aligned Climate"), _own(
        ("Daily LevDAX", "Germany", RV_EURO, "EUR", 0.35, 2),
        ("Daily ShortDAX x2", "Germany", RV_EURO, "EUR", 0.60, 1),
        ("Euro Stoxx Banks", "Eurozone", RV_SECT, "EUR", 0.30, 2),
        ("Green Bond (DR)", "Global", RF_INT, "EUR", 0.25, 1),
        ("MSCI Water ESG Filtered", "Global", RV_SECT, "EUR", 0.60, 1),
        ("New Energy", "Global", RV_SECT, "EUR", 0.60, 1)), currency="EUR", ter_factor=0.9),
    EtfLine("DWS", "Xtrackers", "xtrackers", ("IE", "LU"), 7, 1.0,
            ("ESG", "ESG Screened", "Paris-Aligned Climate", "Equal Weight", "Quality Factor",
             "Value Factor", "Momentum Factor", "Min Vol"), _own(
        ("Artificial Intelligence and Big Data", "Global", RV_SECT, "USD", 0.35, 3),
        ("MSCI World Consumer Staples", "Global", RV_SECT, "USD", 0.25, 1),
        ("Harvest CSI300", "China", RVI_EM, "USD", 0.65, 1)), ter_factor=0.95),
    EtfLine("State Street Global Advisors", "SPDR", "spdr", ("IE",), 5, 0.85,
            ("ESG", "Paris-Aligned Climate", "Equal Weight"), _own(
        ("S&P US Dividend Aristocrats", "United States", RVI_US, "USD", 0.35, 3),
        ("S&P Euro Dividend Aristocrats", "Eurozone", RV_EURO, "EUR", 0.30, 2),
        ("MSCI ACWI IMI", "Global", RVI, "USD", 0.17, 3),
        ("Bloomberg 0-3 Year Euro Corporate Bond", "Eurozone", RF_CP, "EUR", 0.20, 2)), ter_factor=0.85),
    EtfLine("Invesco", "Invesco", "invesco", ("IE",), 5, 0.8, ("ESG", "Equal Weight", "Quality Factor", "Min Vol"), _own(
        ("EQQQ Nasdaq-100", "United States", RVI_US, "USD", 0.30, 4),
        ("CoinShares Global Blockchain", "Global", RV_SECT, "USD", 0.65, 2),
        ("Solar Energy", "Global", RV_SECT, "USD", 0.69, 1),
        ("AT1 Capital Bond", "Global", RF_INT, "USD", 0.39, 1),
        ("BulletShares {year} USD Corporate Bond", "United States", RF_INT, "USD", 0.10, 3)), ter_factor=0.9),
    EtfLine("UBS Asset Management", "UBS ETF", "ubs", ("LU", "IE"), 4, 0.75, ("ESG", "SRI", "Paris-Aligned Climate"), _own(
        ("MSCI Switzerland 20/35", "Switzerland", RVI_EU, "CHF", 0.20, 2),
        ("SMI", "Switzerland", RVI_EU, "CHF", 0.20, 1),
        ("Bloomberg TIPS 1-10", "United States", RF_INT, "USD", 0.20, 1),
        ("MSCI World Socially Responsible", "Global", RVI, "USD", 0.22, 3),
        ("Solactive Global Pure Gold Miners", "Global", RV_SECT, "USD", 0.43, 1))),
    EtfLine("HSBC Asset Management", "HSBC", "ishares", ("IE",), 3, 0.65, ("ESG", "Paris-Aligned Climate", "Min Vol"), _own(
        ("MSCI World Islamic ESG", "Global", RVI, "USD", 0.30, 1),
        ("Hang Seng Tech", "China", RVI_EM, "USD", 0.50, 2),
        ("MSCI Taiwan Capped", "Taiwan", RVI_EM, "USD", 0.50, 1)), ter_factor=0.8),
    EtfLine("JPMorgan Asset Management", "JPM", "jpm_etf", ("IE",), 4, 0.0, (), _own(
        ("US Research Enhanced Index Equity (ESG)", "United States", RVI_US, "USD", 0.20, 5),
        ("Global Research Enhanced Index Equity (ESG)", "Global", RVI, "USD", 0.25, 5),
        ("Europe Research Enhanced Index Equity (ESG)", "Europe", RVI_EU, "EUR", 0.25, 3),
        ("Eurozone Research Enhanced Index Equity (ESG)", "Eurozone", RV_EURO, "EUR", 0.25, 2),
        ("Emerging Markets Research Enhanced Index Equity (ESG)", "Emerging Markets", RVI_EM, "USD", 0.30, 2),
        ("Japan Research Enhanced Index Equity (ESG)", "Japan", RVI_JP, "USD", 0.25, 1),
        ("EUR Ultra-Short Income", "Eurozone", RF_CP, "EUR", 0.18, 2),
        ("USD Ultra-Short Income", "United States", RF_INT, "USD", 0.18, 1),
        ("Global Equity Premium Income", "Global", RVI, "USD", 0.35, 2),
        ("Nasdaq Equity Premium Income", "United States", RVI_US, "USD", 0.35, 2),
        ("BetaBuilders US Treasury Bond 0-1 yr", "United States", RF_INT, "USD", 0.07, 1),
        ("Climate Change Solutions", "Global", RV_SECT, "USD", 0.55, 1))),
    EtfLine("Fidelity International", "Fidelity", "fidelity_etf", ("IE",), 3, 0.0, (), _own(
        ("US Quality Income", "United States", RVI_US, "USD", 0.25, 3),
        ("Global Quality Income", "Global", RVI, "USD", 0.40, 3),
        ("Europe Quality Income", "Europe", RVI_EU, "EUR", 0.30, 2),
        ("Emerging Markets Quality Income", "Emerging Markets", RVI_EM, "USD", 0.50, 1),
        ("Sustainable Research Enhanced US Equity", "United States", RVI_US, "USD", 0.30, 2),
        ("Sustainable Research Enhanced Global Equity", "Global", RVI, "USD", 0.35, 2),
        ("Sustainable Research Enhanced Europe Equity", "Europe", RVI_EU, "EUR", 0.30, 1),
        ("Sustainable Research Enhanced Japan Equity", "Japan", RVI_JP, "USD", 0.30, 1))),
    EtfLine("WisdomTree", "WisdomTree", "plain_etf", ("IE",), 3, 0.25, ("Quality Factor",), _own(
        ("Global Quality Dividend Growth", "Global", RVI, "USD", 0.38, 3),
        ("US Quality Dividend Growth", "United States", RVI_US, "USD", 0.33, 2),
        ("Europe Equity Income", "Europe", RVI_EU, "EUR", 0.29, 1),
        ("Artificial Intelligence", "Global", RV_SECT, "USD", 0.40, 2),
        ("Cybersecurity", "Global", RV_SECT, "USD", 0.45, 2),
        ("Cloud Computing", "Global", RV_SECT, "USD", 0.40, 1),
        ("Battery Solutions", "Global", RV_SECT, "USD", 0.40, 1),
        ("Europe Defence", "Europe", RV_SECT, "EUR", 0.40, 2),
        ("Enhanced Commodity", "Global", MAT, "USD", 0.35, 1)), ter_factor=1.6),
    EtfLine("VanEck", "VanEck", "plain_etf", ("IE",), 3, 0.25, ("Equal Weight",), _own(
        ("Semiconductor", "Global", RV_SECT, "USD", 0.35, 4),
        ("Gold Miners", "Global", RV_SECT, "USD", 0.53, 3),
        ("Junior Gold Miners", "Global", RV_SECT, "USD", 0.55, 1),
        ("Morningstar US Sustainable Wide Moat", "United States", RVI_US, "USD", 0.46, 2),
        ("Morningstar Global Wide Moat", "Global", RVI, "USD", 0.52, 1),
        ("Video Gaming and eSports", "Global", RV_SECT, "USD", 0.55, 1),
        ("Hydrogen Economy", "Global", RV_SECT, "USD", 0.55, 1),
        ("Defense", "Global", RV_SECT, "USD", 0.55, 3),
        ("Rare Earth and Strategic Metals", "Global", RV_SECT, "USD", 0.59, 1),
        ("AEX", "Netherlands", RV_EURO, "EUR", 0.30, 1)), ter_factor=1.8),
    EtfLine("Franklin Templeton", "Franklin", "plain_etf", ("IE",), 2, 0.25, ("Paris-Aligned Climate",), _own(
        ("FTSE India", "India", RVI_EM, "USD", 0.19, 2),
        ("FTSE China", "China", RVI_EM, "USD", 0.19, 1),
        ("FTSE Korea", "South Korea", RVI_EM, "USD", 0.09, 1),
        ("FTSE Brazil", "Brazil", RVI_EM, "USD", 0.19, 1),
        ("Euro Green Bond", "Eurozone", RF_EURO, "EUR", 0.30, 1))),
    EtfLine("Global X ETFs", "Global X", "vanguard", ("IE",), 3, 0.0, (), _own(
        ("Uranium", "Global", RV_SECT, "USD", 0.65, 3),
        ("Lithium & Battery Tech", "Global", RV_SECT, "USD", 0.60, 2),
        ("Robotics & Artificial Intelligence", "Global", RV_SECT, "USD", 0.50, 2),
        ("Copper Miners", "Global", RV_SECT, "USD", 0.65, 2),
        ("Cybersecurity", "Global", RV_SECT, "USD", 0.50, 1),
        ("Hydrogen", "Global", RV_SECT, "USD", 0.50, 1),
        ("Clean Water", "Global", RV_SECT, "USD", 0.50, 1),
        ("Defence Tech", "Global", RV_SECT, "USD", 0.50, 2),
        ("Silver Miners", "Global", RV_SECT, "USD", 0.65, 1),
        ("S&P 500 Covered Call", "United States", RVI_US, "USD", 0.45, 2),
        ("Nasdaq 100 Covered Call", "United States", RVI_US, "USD", 0.45, 2),
        ("Disruptive Materials", "Global", RV_SECT, "USD", 0.50, 1))),
)

# --- mutual funds -----------------------------------------------------------------------
Letters = tuple[tuple[str, float, float], ...]  # (class letter, popularity, TER factor)


@dataclass(frozen=True, slots=True)
class FundRange:
    """A Luxembourg SICAV umbrella whose sub-funds are drawn from STRATEGIES."""

    company: str
    brand: str  # "A|B" picks one per sub-fund; "{asset}" becomes Equity / Bond / Strategy
    sep: str
    style: str
    weight: float
    letters: Letters
    suffix: str  # "Fund" when the house appends it to sub-fund names
    size: int
    ter_factor: float = 1.0


FUND_RANGES = (
    FundRange("BlackRock", "BlackRock Global Funds", " - ", "bgf", 5,
              (("A", 1, 1), ("D", .45, .6), ("E", .3, 1.3), ("I", .25, .45)), "Fund", 48),
    FundRange("JPMorgan Asset Management", "JPMorgan Funds", " - ", "jpm", 5,
              (("A", 1, 1), ("C", .4, .55), ("D", .35, 1.35), ("I", .25, .45), ("T", .15, 1.35)), "Fund", 48),
    FundRange("Fidelity International", "Fidelity Funds", " - ", "fidelity", 5,
              (("A", 1, 1), ("E", .35, 1.3), ("Y", .4, .55), ("I", .2, .45), ("W", .1, .55)), "Fund", 48),
    FundRange("Amundi", "Amundi Funds", " - ", "amundi_fund", 4,
              (("A", 1, 1), ("E", .4, 1.1), ("F", .3, 1.4), ("I", .2, .45), ("R", .2, .6)), "", 42, 0.95),
    FundRange("DWS", "DWS Invest", " ", "dws", 3,
              (("L", 1, 1), ("N", .4, 1.3), ("F", .3, .55), ("TF", .2, .6), ("I", .15, .45)), "", 38, 0.9),
    FundRange("Invesco", "Invesco Funds", " - ", "invesco_fund", 3,
              (("A", 1, 1), ("C", .35, .6), ("E", .4, 1.3), ("Z", .25, .55)), "Fund", 38),
    FundRange("UBS Asset Management", "UBS (Lux) {asset} Fund", " - ", "ubs_fund", 3,
              (("P", 1, 1), ("Q", .5, .6), ("I-A1", .2, .45), ("K-1", .1, .5)), "", 38, 0.95),
    FundRange("HSBC Asset Management", "HSBC Global Investment Funds", " - ", "hsbc_fund", 2,
              (("A", 1, 1), ("E", .3, 1.3), ("I", .2, .45), ("X", .1, .5)), "", 32, 0.9),
    FundRange("Franklin Templeton", "Templeton|Franklin", " ", "templeton", 3,
              (("A", 1, 1), ("N", .4, 1.4), ("I", .2, .5), ("W", .2, .6)), "Fund", 40, 1.05),
)

# adjective, region, category, base currency, popularity
EQUITY_REGIONS = (
    ("Global", "Global", RVI, "USD", 5), ("World", "Global", RVI, "USD", 2),
    ("European", "Europe", RVI_EU, "EUR", 4), ("Euroland", "Eurozone", RV_EURO, "EUR", 2),
    ("US", "United States", RVI_US, "USD", 4), ("America", "United States", RVI_US, "USD", 1),
    ("Emerging Markets", "Emerging Markets", RVI_EM, "USD", 3), ("Asian", "Asia Pacific", RVI_EM, "USD", 2),
    ("Japan", "Japan", RVI_JP, "JPY", 2), ("China", "China", RVI_EM, "USD", 2),
    ("India", "India", RVI_EM, "USD", 1), ("Latin American", "Latin America", RVI_EM, "USD", 1),
    ("Nordic", "Europe", RVI_EU, "EUR", 1), ("Swiss", "Switzerland", RVI_EU, "CHF", 1),
    ("UK", "United Kingdom", RVI_EU, "GBP", 1), ("German", "Germany", RV_EURO, "EUR", 1),
)
EQUITY_STYLES = (
    ("Equity", 4), ("Equity Income", 2), ("Growth", 2), ("Value", 2), ("Smaller Companies", 1.5),
    ("Sustainable Equity", 2), ("Dividend", 1.5), ("Focus", 1), ("Opportunities", 1.5), ("Select", 1),
    ("Special Situations", 1), ("Leaders", 0.8), ("Quality", 0.8),
)
THEMES = (
    ("Technology", 3), ("Healthcare", 2), ("Sustainable Energy", 2), ("Infrastructure", 1.5),
    ("Real Estate Securities", 1), ("Water", 1), ("Natural Resources", 1), ("Gold and Precious Metals", 1),
    ("Consumer Brands", 0.8), ("Financials", 0.8), ("Biotechnology", 0.8), ("Climate Change", 1),
    ("Artificial Intelligence", 1.5), ("Robotics", 1), ("Digital Economy", 0.8), ("Future Health", 0.6),
    ("Circular Economy", 0.6), ("Smart Mobility", 0.6),
)
BOND_REGIONS = (
    ("Global", "Global", RF_INT, "USD", 3), ("Euro", "Eurozone", RF_EURO, "EUR", 3),
    ("US Dollar", "United States", RF_INT, "USD", 2), ("Emerging Markets", "Emerging Markets", RF_INT, "USD", 2),
    ("European", "Europe", RF_EURO, "EUR", 1), ("Asian", "Asia Pacific", RF_INT, "USD", 1),
)
BOND_STYLES = (
    ("Bond", 2), ("Corporate Bond", 2), ("High Yield Bond", 2), ("Short Duration Bond", 1.5),
    ("Government Bond", 1.5), ("Aggregate Bond", 1), ("Total Return Bond", 1), ("Strategic Bond", 1),
    ("Inflation Linked Bond", 0.8), ("Flexible Bond", 0.8), ("Green Bond", 0.8), ("Income", 1),
)
OTHER_STRATEGIES = (  # name, category, base currency, popularity
    ("Global Multi-Asset Income", MIX_M, "USD", 2), ("Global Allocation", MIX_M, "USD", 2),
    ("Balanced", MIX_M, "EUR", 1.5), ("Conservative Allocation", MIX_C, "EUR", 1.5),
    ("Dynamic Allocation", MIX_A, "EUR", 1.2), ("Flexible Allocation", MIX_M, "EUR", 1),
    ("Income Builder", MIX_M, "USD", 1), ("Defensive", MIX_C, "EUR", 0.8),
    ("Euro Liquidity", MONET, "EUR", 1), ("US Dollar Liquidity", MONET, "USD", 0.6),
)


@dataclass(frozen=True, slots=True)
class FundList:
    """Explicitly named funds of one brand: index-fund ranges and Spanish FIs."""

    company: str
    brand: str
    style: str
    domicile: str
    kind: str  # "Index fund" or "FI"
    weight: float
    letters: Letters
    funds: tuple[tuple[str, str], ...]  # (body, COMMON exposure name or category); "{year}" = series


def _fi(company: str, brand: str, weight: float, letters: Letters, *funds: tuple[str, str]) -> FundList:
    return FundList(company, brand, "fi", "ES", "FI", weight, letters, funds)


INDEX_FUNDS = (
    FundList("Vanguard", "Vanguard", "vanguard_fund", "IE", "Index fund", 4,
             (("", 1, 1), ("Investor", .6, 1.6), ("Institutional Plus", .3, .5)), (
        ("Global Stock Index Fund", "MSCI World"), ("U.S. 500 Stock Index Fund", "S&P 500"),
        ("Emerging Markets Stock Index Fund", "MSCI Emerging Markets"), ("Eurozone Stock Index Fund", "MSCI EMU"),
        ("European Stock Index Fund", "MSCI Europe"), ("Japan Stock Index Fund", "MSCI Japan"),
        ("Pacific ex-Japan Stock Index Fund", "MSCI Pacific ex Japan"),
        ("Global Small-Cap Index Fund", "MSCI World Small Cap"), ("Global Bond Index Fund", "Global Aggregate Bond"),
        ("Euro Government Bond Index Fund", "EUR Govt Bond"), ("U.S. Government Bond Index Fund", "USD Treasury Bond"),
        ("ESG Developed World All Cap Equity Index Fund", "MSCI World"),
        ("LifeStrategy 20% Equity Fund", MIX_C), ("LifeStrategy 40% Equity Fund", MIX_C),
        ("LifeStrategy 60% Equity Fund", MIX_M), ("LifeStrategy 80% Equity Fund", MIX_A))),
    FundList("BlackRock", "iShares", "ishares_fund", "IE", "Index fund", 4,
             (("D", 1, 1), ("Flex", .4, .6), ("Inst", .3, .7), ("S", .2, .8)), (
        ("Developed World Index Fund (IE)", "MSCI World"), ("Emerging Markets Index Fund (IE)", "MSCI Emerging Markets"),
        ("North America Index Fund (IE)", "MSCI USA"), ("Europe Index Fund (IE)", "MSCI Europe"),
        ("Japan Index Fund (IE)", "MSCI Japan"), ("Pacific Index Fund (IE)", "MSCI Pacific ex Japan"),
        ("Euro Government Bond Index Fund (IE)", "EUR Govt Bond"),
        ("Global Government Bond Index Fund (IE)", "Global Aggregate Bond"),
        ("ESG Screened Developed World Index Fund (IE)", "MSCI World"),
        ("Developed Real Estate Index Fund (IE)", "FTSE EPRA Nareit Developed"))),
    FundList("Fidelity International", "Fidelity", "fidelity", "IE", "Index fund", 3, (("P", 1, 1),), (
        ("MSCI World Index Fund", "MSCI World"), ("S&P 500 Index Fund", "S&P 500"),
        ("MSCI Europe Index Fund", "MSCI Europe"), ("MSCI Emerging Markets Index Fund", "MSCI Emerging Markets"),
        ("MSCI Japan Index Fund", "MSCI Japan"), ("MSCI Pacific ex-Japan Index Fund", "MSCI Pacific ex Japan"))),
    FundList("Amundi", "Amundi", "amundi_index", "LU", "Index fund", 3, (("A", 1, 1), ("I", .3, .5), ("R", .3, .7)), (
        ("Index MSCI World", "MSCI World"), ("Index S&P 500", "S&P 500"),
        ("Index MSCI Emerging Markets", "MSCI Emerging Markets"), ("Index MSCI Europe", "MSCI Europe"),
        ("Index MSCI EMU", "MSCI EMU"), ("Index MSCI Japan", "MSCI Japan"),
        ("Index Euro Corporate SRI", "EUR Corporate Bond"), ("Index Global Aggregate", "Global Aggregate Bond"))),
)

SPANISH_FUNDS = (
    _fi("Santander Asset Management", "Santander", 5, (("A", 1, 1), ("B", .5, .8), ("I", .3, .5), ("Cartera", .3, .4)),
        ("Renta Fija Corto Plazo FI", RF_CP), ("Acciones Españolas FI", RV_NAC), ("Small Caps España FI", RV_NAC),
        ("Tándem 20-60 FI", MIX_M), ("Sostenible Bonos FI", RF_EURO), ("Responsabilidad Conservador FI", MIX_C),
        ("Dividendo Europa FI", RVI_EU), ("Select Moderado FI", MIX_M), ("Select Dinámico FI", MIX_A),
        ("Select Defensivo FI", MIX_C), ("Renta Fija Horizonte {year} FI", RENT_OBJ)),
    _fi("Bestinver Gestión", "Bestinver", 4, (("", 1, 1),),
        ("Internacional FI", RVI), ("Bolsa FI", RV_NAC), ("Grandes Compañías FI", RVI), ("Patrimonio FI", MIX_C),
        ("Renta FI", RF_CP), ("Mixto FI", MIX_M), ("Latam FI", RVI_EM)),
    _fi("Cobas Asset Management", "Cobas", 4, (("C", 1, 1), ("D", .4, .8), ("P", .3, .6)),
        ("Selección FI", RVI), ("Internacional FI", RVI), ("Iberia FI", RV_NAC), ("Grandes Compañías FI", RVI),
        ("Renta FI", RF_CP), ("Concentrados FIL", RVI)),
    _fi("CaixaBank Asset Management", "CaixaBank", 5,
        (("Estándar", 1, 1), ("Plus", .5, .8), ("Premium", .4, .7), ("Platinum", .3, .5), ("Cartera", .3, .4),
         ("Universal", .4, 1.1)),
        ("Bolsa Selección Europa FI", RVI_EU), ("Bolsa USA FI", RVI_US), ("Bolsa Índice España FI", RV_NAC),
        ("Bolsa Selección Emergentes FI", RVI_EM), ("Bolsa Gestión España FI", RV_NAC),
        ("Master Renta Fija Deuda Pública 1-3 FI", RF_CP), ("Monetario Rendimiento FI", MONET),
        ("Selección Futuro Sostenible FI", RVI), ("Selección Tendencias FI", RV_SECT),
        ("Renta Fija Horizonte {year} FI", RENT_OBJ)),
    _fi("BBVA Asset Management", "BBVA", 5, (("", 1, 1), ("A", .5, .9), ("B", .3, .6)),
        ("Bonos Corto Plazo FI", RF_CP), ("Bolsa Europa FI", RVI_EU), ("Bolsa USA FI", RVI_US),
        ("Bolsa Desarrollo Sostenible ISR FI", RVI), ("Quality Inversión Moderada FI", MIX_M),
        ("Quality Inversión Conservadora FI", MIX_C), ("Bolsa Tecnología y Telecomunicaciones FI", RV_SECT),
        ("Bolsa Índice Euro FI", RV_EURO), ("Bonos Horizonte {year} FI", RENT_OBJ)),
    _fi("Magallanes Value Investors", "Magallanes", 3, (("M", 1, 1), ("P", .5, .7), ("E", .3, .6), ("I", .2, .5)),
        ("European Equity FI", RVI_EU), ("Iberian Equity FI", RV_NAC), ("Microcaps Europe FI", RVI_EU)),
    _fi("azValor Asset Management", "azValor", 3, (("", 1, 1),),
        ("Internacional FI", RVI), ("Iberia FI", RV_NAC), ("Managers FI", RVI), ("Blue Chips FI", RVI)),
    _fi("Bankinter Gestión de Activos", "Bankinter", 4, (("R", 1, 1), ("C", .5, .6), ("I", .3, .4)),
        ("Índice Europeo 50 FI", RV_EURO), ("Índice América FI", RVI_US), ("Índice Japón FI", RVI_JP),
        ("Índice Emergentes FI", RVI_EM), ("Renta Fija Largo Plazo FI", RF_EURO), ("Dividendo Europa FI", RVI_EU),
        ("Bolsa España FI", RV_NAC), ("Monetario FI", MONET), ("Sostenibilidad FI", MIX_M)),
    _fi("Renta 4 Gestora", "Renta 4", 3, (("", 1, 1), ("R", .5, 1), ("I", .3, .5)),
        ("Bolsa FI", RV_NAC), ("Renta Fija FI", RF_EURO), ("Valor Europa FI", RVI_EU), ("Monetario FI", MONET),
        ("Nexus FI", MIX_M), ("Megatendencias Tecnología FI", RV_SECT), ("Renta Fija 6 Meses FI", RF_CP)),
    _fi("Ibercaja Gestión", "Ibercaja", 4, (("A", 1, 1), ("B", .5, .8), ("C", .3, .6), ("E", .2, .5), ("I", .2, .4)),
        ("Gestión Crecimiento FI", MIX_A), ("Gestión Equilibrada FI", MIX_M), ("Gestión Audaz FI", MIX_A),
        ("Bolsa Internacional FI", RVI), ("Tecnológico FI", RV_SECT), ("Sanidad FI", RV_SECT),
        ("Ahorro Renta Fija FI", RF_CP), ("Bolsa España FI", RV_NAC), ("Renta Fija {year} FI", RENT_OBJ)),
    _fi("Mutuactivos", "Mutuafondo", 3, (("A", 1, 1), ("D", .4, .7), ("L", .3, .5), ("F", .2, .4)),
        ("FI", RF_CP), ("Corto Plazo FI", RF_CP), ("Bonos Flotantes FI", RF_CP), ("España FI", RV_NAC),
        ("Dinero FI", MONET), ("Renta Fija Flexible FI", RF_EURO)),
    _fi("Kutxabank Gestión", "Kutxabank", 3, (("Estándar", 1, 1), ("Cartera", .4, .4), ("Plus", .3, .7)),
        ("Bolsa Internacional FI", RVI), ("Renta Fija Corto Plazo FI", RF_CP), ("Bolsa Eurozona FI", RV_EURO),
        ("Bono FI", RF_EURO), ("Dividendo FI", RVI_EU), ("Bolsa Emergentes FI", RVI_EM), ("Bolsa USA FI", RVI_US)),
)

# How the brand is sometimes written in names (probabilities; the rest uses the main spelling).
BRAND_VARIANTS = {
    "iShares": (("ISHARES", .05), ("ISHS", .04), ("iSh", .03)),
    "Vanguard": (("VANGUARD", .05), ("Vang.", .04)),
    "Amundi": (("AMUNDI", .05),),
    "Lyxor": (("LYXOR", .06),),
    "Xtrackers": (("X-trackers", .05), ("Xtr.", .04), ("XTRACKERS", .03)),
    "Invesco": (("INVESCO", .05),),
    "JPM": (("J.P. Morgan", .08), ("JPMorgan", .08)),
    "Fidelity": (("FIDELITY", .05), ("Fid.", .03)),
    "WisdomTree": (("WISDOMTREE", .05), ("Wisdom Tree", .04)),
    "VanEck": (("Van Eck", .06), ("VANECK", .04)),
    "Franklin": (("FRANKLIN", .05),),
    "Templeton": (("TEMPLETON", .05),),
    "Global X": (("GLOBAL X", .05), ("GlobalX", .03)),
    "BlackRock Global Funds": (("BGF", .06), ("BlackRock GF", .04)),
    "JPMorgan Funds": (("JPM Funds", .08), ("JPMORGAN FUNDS", .04)),
    "Fidelity Funds": (("FF", .05), ("Fidelity Fds", .04)),
    "Amundi Funds": (("AMUNDI FUNDS", .05),),
    "DWS Invest": (("DWS INVEST", .05),),
    "Invesco Funds": (("Invesco Fds", .05),),
    "HSBC Global Investment Funds": (("HSBC GIF", .10),),
    "Santander": (("SANTANDER", .05),),
    "Bestinver": (("BESTINVER", .05),),
    "Cobas": (("COBAS", .05),),
    "CaixaBank": (("Caixabank", .06), ("CAIXABANK", .04)),
    "Magallanes": (("MAGALLANES", .05),),
    "azValor": (("Azvalor", .06), ("AZVALOR", .04)),
    "Bankinter": (("BANKINTER", .05),),
    "Renta 4": (("RENTA 4", .05), ("R4", .03)),
    "Ibercaja": (("IBERCAJA", .05),),
    "Mutuafondo": (("MUTUAFONDO", .05),),
    "Kutxabank": (("KUTXABANK", .05), ("Kutxa", .03)),
}
# Abbreviations people type for a brand word (query side; tools/evaluate.py uses them).
ABBREVIATIONS = {
    "iShares": ("ISHS", "iSh"), "Vanguard": ("Vang.", "VANG"), "Xtrackers": ("Xtr.", "XTRACK"),
    "Lyxor": ("LYX",), "Invesco": ("Invsc",), "Fidelity": ("Fid.", "FIDEL"), "WisdomTree": ("WisdomTr.",),
    "Franklin": ("Frank.",), "Templeton": ("Templ.",), "JPMorgan": ("JPM",), "Santander": ("Sant.",),
    "Bestinver": ("Bestinv.",), "CaixaBank": ("Caixa",), "Bankinter": ("Bankint.",), "Ibercaja": ("Iberc.",),
    "Kutxabank": ("Kutxa",), "Magallanes": ("Magall.",), "Mutuafondo": ("Mutuaf.",),
}
# Short words people type in queries: a ticker spelled like one would hijack them.
COMMON_WORDS = frozenset(
    "and the for from with all one new top big low high best fund bond gold world index usa "
    "del las los con por para una uno fondo bonos oro plata bolsa renta fija mundo eeuu euro".split()
)
P_BRAND_ABSENT = 0.04  # name written without its brand
P_UPPERCASE = 0.03  # whole name in capitals
P_NO_ACCENTS = 0.2  # Spanish names that lost their accents

# --- share classes ----------------------------------------------------------------------
_XTR_SERIES = {"EUR": 2, "GBP": 3, "CHF": 4, "USD": 5, "JPY": 6, "MXN": 7, "SEK": 8}


def _ad(acc: bool, accumulating: str, distributing: str) -> str:
    return accumulating if acc else distributing


def _foreign(currency: str, base: str, before: str = "", after: str = " ") -> str:
    """Currency mention for a non-hedged class in a currency other than the fund's base."""
    return "" if currency == base else f"{before}{currency}{after}"


# style -> label(letter, currency, accumulating, hedged, base currency)
CLASS_LABELS: dict[str, Callable[[str, str, bool, bool, str], str]] = {
    "ishares": lambda l, c, a, h, b: f"{c}{' Hedged' if h else ''} ({_ad(a, 'Acc', 'Dist')})",
    "vanguard": lambda l, c, a, h, b: f"{c}{' Hedged' if h else ''} {_ad(a, 'Accumulating', 'Distributing')}",
    "amundi": lambda l, c, a, h, b: f"{c}{' Hedged' if h else ''} ({_ad(a, 'C', 'D')})",
    "lyxor": lambda l, c, a, h, b: "- " + (f"Daily Hedged to {c} - " if h else _foreign(c, b))
    + _ad(a, "Acc", "Dist"),
    "xtrackers": lambda l, c, a, h, b: f"{_XTR_SERIES.get(c, 9) if h else 1}{_ad(a, 'C', 'D')}"
    + (f" - {c} Hedged" if h else _foreign(c, b, " ", "")),
    "spdr": lambda l, c, a, h, b: (f"{c} Hdg " if h else _foreign(c, b)) + f"({_ad(a, 'Acc', 'Dist')})",
    "invesco": lambda l, c, a, h, b: (f"{c} Hdg " if h else _foreign(c, b)) + _ad(a, "Acc", "Dist"),
    "ubs": lambda l, c, a, h, b: f"({'hedged to ' + c if h else c}) A-{_ad(a, 'acc', 'dis')}",
    "jpm_etf": lambda l, c, a, h, b: f"- {c}{' Hdg' if h else ''} ({_ad(a, 'acc', 'dist')})",
    "fidelity_etf": lambda l, c, a, h, b: f"{c}{' Hedged' if h else ''} {_ad(a, 'Acc', 'Inc')}",
    "plain_etf": lambda l, c, a, h, b: f"{c}{' Hedged' if h else ''} {_ad(a, 'Acc', 'Dist')}",
    "etc": lambda l, c, a, h, b: f"{c} Hedged" if h else "",
    "jpm": lambda l, c, a, h, b: f"{l} ({_ad(a, 'acc', 'dist')}) - {c}{' (hedged)' if h else ''}",
    "fidelity": lambda l, c, a, h, b: f"{l}-{_ad(a, 'Acc', 'Dis')}-{c}{' (hedged)' if h else ''}",
    "bgf": lambda l, c, a, h, b: f"{l}{_ad(a, '2', '4')} {c}{' Hedged' if h else ''}",
    "amundi_fund": lambda l, c, a, h, b: f"{l} {c}{' Hdg' if h else ''} ({_ad(a, 'C', 'D')})",
    "dws": lambda l, c, a, h, b: _foreign(c, b) + f"{l}{_ad(a, 'C', 'D')}{'H' if h else ''}",
    "invesco_fund": lambda l, c, a, h, b: f"{l} {c}{' Hdg' if h else ''} {_ad(a, 'Acc', 'Dist')}",
    "ubs_fund": lambda l, c, a, h, b: f"({c}{' hedged' if h else ''}) {l}-{_ad(a, 'acc', 'dist')}",
    "hsbc_fund": lambda l, c, a, h, b: f"{l}{_ad(a, 'C', 'D')}{'H' if h else ''}{'' if c == b else c}",
    "templeton": lambda l, c, a, h, b: f"{l} ({_ad(a, 'acc', 'Ydis')}) {c}{'-H1' if h else ''}",
    "vanguard_fund": lambda l, c, a, h, b: f"{l + ' ' if l else ''}{c}{' Hedged' if h else ''} {_ad(a, 'Acc', 'Dist')}",
    "ishares_fund": lambda l, c, a, h, b: f"{l} {_ad(a, 'Acc', 'Dist')} {c}{' Hedged' if h else ''}",
    "amundi_index": lambda l, c, a, h, b: f"{l}{'H' if h else ''}{c[0]}-{_ad(a, 'C', 'D')}",
    "fi": lambda l, c, a, h, b: f"Clase {l}" if l else "",
}
# family -> currencies offered besides the base one: unhedged and hedged (currency, popularity)
UNHEDGED = {
    "etf": (("EUR", .12), ("USD", .1), ("GBP", .06)),
    "fund": (("EUR", .9), ("USD", .6), ("GBP", .1), ("CHF", .08), ("SGD", .03), ("HKD", .03)),
    "index": (("USD", .5), ("GBP", .1)),
}
HEDGED = {
    "etf": (("EUR", .45), ("USD", .25), ("GBP", .18), ("CHF", .15), ("JPY", .02), ("MXN", .015), ("SEK", .015)),
    "fund": (("EUR", .5), ("USD", .25), ("CHF", .12), ("GBP", .1), ("SEK", .02), ("PLN", .02), ("CZK", .015),
             ("HUF", .015), ("SGD", .015), ("AUD", .015)),
    "index": (("USD", .1), ("GBP", .1), ("CHF", .1)),
    "etc": (("EUR", .4), ("GBP", .15), ("CHF", .1)),
}
DIST_WEIGHT = {"etf": .55, "mm": .5, "fund": .4, "index": .3}
# Share of the rows drawn from each kind of product (capped by what exists).
GROUP_SHARES = {"ETF": 0.45, "Fund": 0.35, "Index fund": 0.07, "FI": 0.13}
LOG_AUM = {"ETF": 5.0, "ETC": 6.0, "Fund": 5.3, "Index fund": 5.8, "FI": 4.0}  # mean log(M EUR)


@dataclass(frozen=True, slots=True)
class _Share:
    label: str
    currency: str
    acc: bool
    hedged: bool
    weight: float
    ter_factor: float


@dataclass(frozen=True, slots=True)
class _Fund:
    company: str
    brand: str
    sep: str
    body: str
    short_name: str
    kind: str  # ETF, ETC, Fund, Index fund, FI
    region: str
    category: str
    currency: str  # base currency
    domicile: str
    style: str
    family: str  # share-class family: etf, mm, etc, fund, index, fi
    letters: Letters
    ter: float
    weight: float
    launch: int


@dataclass(slots=True)
class FundRecord:
    """One generated row plus the pieces its fund name is made of (used by tools/evaluate.py)."""

    cells: list[str]
    company: str
    brand: str  # main spelling of the product line's brand: "iShares", "Lyxor", "JPMorgan Funds"
    brand_text: str  # the brand as written in this row ('' when the name omits it)
    sep: str  # between brand and body: " " or " - "
    body: str  # between brand and share class: "Core S&P 500 UCITS ETF"
    share_class: str  # "USD (Acc)", "Clase C", '' for single-class funds
    short_name: str  # distinctive part: "Core S&P 500", "European Value", "Internacional"
    kind: str  # ETF, ETC, Fund, Index fund or FI
    upper: bool  # the whole name is written in capitals

    @property
    def name(self) -> str:
        return self.cells[2]


def compose_name(brand: str, sep: str, body: str, share_class: str) -> str:
    """``brand + sep + body + ' ' + share_class``, skipping empty parts."""
    name = f"{brand}{sep}{body}" if brand else body
    return f"{name} {share_class}" if share_class else name


# --- catalog ----------------------------------------------------------------------------
def _jitter(rng: random.Random, sigma: float = 0.35) -> float:
    return math.exp(rng.gauss(0.0, sigma))


def _expand_years(exposures: tuple[Exposure, ...]) -> list[Exposure]:
    """Replace each "{year}" exposure by a target-maturity series, one exposure per year."""
    out = []
    for exposure in exposures:
        if "{year}" not in exposure.name:
            out.append(exposure)
            continue
        out.extend(
            Exposure(exposure.name.format(year=year), exposure.region, exposure.category, exposure.currency,
                     exposure.ter, exposure.popularity / 3, exposure.group)
            for year in range(2025, 2033)
        )
    return out


def _etf(line: EtfLine, exposure: Exposure, modifier: str | None, rng: random.Random,
         tag: str = "", domicile: str = "") -> _Fund:
    name = exposure.name
    popularity, ter_factor = 1.0, 1.0
    if modifier:
        position, popularity, ter_factor = MODIFIERS[modifier]
        if position == "prefix" or (modifier == "ESG" and line.brand == "Xtrackers"):
            name = f"{modifier} {name}"
        else:
            name = f"{name} {modifier}"
    asset = CATEGORIES[exposure.category][0]
    if line.brand == "Lyxor" and asset == "Equity" and not name.startswith("Daily"):
        name += " (DR)"  # Lyxor marks physically replicated equity ETFs
    if tag:
        name += f" {tag}"
    etc = exposure.name.startswith("Physical ")
    body = f"{name} {'ETC' if etc else 'UCITS ETF'}" + (" (DE)" if domicile == "DE" else "")
    if not domicile:
        domicile = "JE" if etc and line.brand == "WisdomTree" else rng.choice(line.domiciles)
    brand = line.brand
    if brand == "UBS ETF" and domicile == "IE":
        brand = "UBS (Irl) ETF"
    family = "etc" if etc else "mm" if asset == "Money Market" else "etf"
    return _Fund(
        company=line.company, brand=brand, sep=" ", body=body, short_name=name.replace(" (DR)", ""),
        kind="ETC" if etc else "ETF", region=exposure.region, category=exposure.category,
        currency=line.currency or exposure.currency, domicile=domicile, style="etc" if etc else line.style,
        family=family, letters=(("", 1.0, 1.0),),
        ter=max(0.03, round(exposure.ter * line.ter_factor * ter_factor * rng.uniform(0.85, 1.15), 2)),
        weight=line.weight * exposure.popularity * popularity * (0.5 if tag or domicile == "DE" else 1)
        * _jitter(rng),
        launch=max(2001, 2024 - int(rng.random() * (6 + 1.3 * exposure.popularity))),
    )


def _etf_funds(rng: random.Random) -> list[_Fund]:
    funds: list[_Fund] = []
    for line in ETF_LINES:
        for exposure in COMMON:
            if line.covers:
                if exposure.name not in line.covers:
                    continue
            elif rng.random() >= min(1.0, line.breadth * (0.3 + 0.07 * exposure.popularity)):
                continue
            usable = [m for m in GROUP_MODIFIERS[exposure.group] if m in line.modifiers]
            variants = [None] + [m for m in usable if rng.random() < 0.3 + 0.4 * MODIFIERS[m][1]]
            equity = CATEGORIES[exposure.category][0] == "Equity"
            for modifier in variants:
                funds.append(_etf(line, exposure, modifier, rng))
                if modifier is None and line.brand == "Xtrackers" and exposure.group == "broad" and rng.random() < .5:
                    funds.append(_etf(line, exposure, None, rng, tag="Swap"))
                if (line.brand == "iShares" and modifier in (None, "Core") and equity
                        and exposure.currency == "EUR" and rng.random() < .5):
                    funds.append(_etf(line, exposure, modifier, rng, domicile="DE"))
        funds.extend(_etf(line, exposure, None, rng) for exposure in _expand_years(line.own))
    return funds


def _strategies() -> list[tuple[str, str, str, str, float]]:
    """(name, region, category, base currency, popularity) of every SICAV sub-fund idea."""
    out = []
    for adjective, region, category, currency, weight in EQUITY_REGIONS:
        out.extend((f"{adjective} {style}", region, category, currency, weight * pop) for style, pop in EQUITY_STYLES)
    for theme, pop in THEMES:
        out.append((theme, "Global", RV_SECT, "USD", 1.5 * pop))
        out.append((f"Global {theme}", "Global", RV_SECT, "USD", 1.2 * pop))
    for adjective, region, category, currency, weight in BOND_REGIONS:
        for style, pop in BOND_STYLES:
            short = style == "Short Duration Bond" and category == RF_EURO
            out.append((f"{adjective} {style}", region, RF_CP if short else category, currency, weight * pop))
    out.extend((name, "Global", category, currency, 3 * pop) for name, category, currency, pop in OTHER_STRATEGIES)
    return out


def _weighted_sample(rng: random.Random, items: list, weights: list[float], k: int) -> list:
    """k distinct items, drawn with probability proportional to weight (Efraimidis-Spirakis)."""
    keyed = [(math.log(rng.random() or 1e-300) / w, i) for i, w in enumerate(weights)]
    return [items[i] for _, i in heapq.nlargest(k, keyed)]


def _range_funds(rng: random.Random) -> list[_Fund]:
    strategies = _strategies()
    funds = []
    for fund_range in FUND_RANGES:
        picked = _weighted_sample(rng, strategies, [s[4] for s in strategies], fund_range.size)
        for name, region, category, currency, popularity in picked:
            asset = CATEGORIES[category][0]
            brand = rng.choice(fund_range.brand.split("|")).format(
                asset={"Equity": "Equity", "Fixed Income": "Bond"}.get(asset, "Strategy"))
            body = f"{name} {fund_range.suffix}".strip()
            funds.append(_Fund(
                company=fund_range.company, brand=brand, sep=fund_range.sep, body=body, short_name=name,
                kind="Fund", region=region, category=category, currency=currency, domicile="LU",
                style=fund_range.style, family="fund", letters=fund_range.letters,
                ter=round(CATEGORIES[category][2] * fund_range.ter_factor * rng.uniform(0.85, 1.15), 2),
                weight=fund_range.weight * popularity * _jitter(rng), launch=rng.randint(1995, 2022),
            ))
    return funds


def _list_funds(rng: random.Random) -> list[_Fund]:
    funds = []
    for fund_list in INDEX_FUNDS + SPANISH_FUNDS:
        entries = []
        for body, key in fund_list.funds:  # "{year}" bodies are target-maturity series
            years = range(2026, 2032) if "{year}" in body else (0,)
            entries.extend((body.format(year=year), key, year) for year in years)
        for body, key, year in entries:
            exposure = COMMON_BY_NAME.get(key)
            if exposure is not None:
                region, category, ter = exposure.region, exposure.category, exposure.ter * 1.3
            else:
                region, category, ter = CATEGORIES[key][1], key, CATEGORIES[key][2]
                if fund_list.kind == "Index fund":
                    ter *= 0.2
            short = " ".join(word for word in body.split() if word not in ("FI", "FIL", "(IE)"))
            if year:
                launch = min(2025, year - rng.randint(2, 4))
            else:
                launch = rng.randint(1988, 2023) if fund_list.kind == "FI" else rng.randint(1998, 2021)
            funds.append(_Fund(
                company=fund_list.company, brand=fund_list.brand, sep=" ", body=body,
                short_name=short or fund_list.brand, kind=fund_list.kind, region=region, category=category,
                currency="EUR", domicile=fund_list.domicile, style=fund_list.style,
                family="fi" if fund_list.kind == "FI" else "index", letters=fund_list.letters,
                ter=max(0.05, round(ter * rng.uniform(0.85, 1.15), 2)),
                weight=fund_list.weight * (0.35 if year else 1.0) * _jitter(rng), launch=launch,
            ))
    return funds


@functools.lru_cache(maxsize=None)
def _share_classes(style: str, family: str, letters: Letters, base: str) -> tuple[_Share, ...]:
    """Every share class a fund of this style could have; the first one is the main class."""
    currencies = [(base, 1.0, False)]
    currencies += [(c, w, False) for c, w in UNHEDGED.get(family, ()) if c != base]
    currencies += [(c, w, True) for c, w in HEDGED.get(family, ()) if c != base]
    label = CLASS_LABELS[style]
    shares = []
    for letter, letter_weight, letter_ter in letters:
        for currency, currency_weight, hedged in currencies:
            for acc in (True, False):
                dist_weight = 1.0 if acc else DIST_WEIGHT.get(family, 0.0)
                if dist_weight:
                    shares.append(_Share(
                        label(letter, currency, acc, hedged, base), currency, acc, hedged,
                        letter_weight * currency_weight * dist_weight, letter_ter * (1.03 if hedged else 1.0),
                    ))
    return tuple(shares)


def _catalog(rng: random.Random) -> list[_Fund]:
    return _etf_funds(rng) + _range_funds(rng) + _list_funds(rng)


def _group(kind: str) -> str:
    return "ETF" if kind == "ETC" else kind


def _quotas(rows: int, sizes: dict[str, int]) -> dict[str, int]:
    """Split ``rows`` across groups by GROUP_SHARES, giving what a full group cannot take to the others."""
    if rows > sum(sizes.values()):
        raise ValueError(f"at most {sum(sizes.values())} unique rows can be generated, asked for {rows}")
    quotas = dict.fromkeys(sizes, 0)
    remaining = rows
    while remaining:
        open_groups = [g for g in sizes if quotas[g] < sizes[g]]
        total = sum(GROUP_SHARES[g] for g in open_groups)
        budget = remaining
        for group in open_groups:
            want = max(1, math.floor(budget * GROUP_SHARES[group] / total))
            take = min(sizes[group] - quotas[group], remaining, want)
            quotas[group] += take
            remaining -= take
    return quotas


# --- identifiers ------------------------------------------------------------------------
_SEDOL = "0123456789BCDFGHJKLMNPQRSTVWXYZ"
_ALNUM = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
_DIGITS = "0123456789"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def isin_check_digit(body: str) -> str:
    """ISO 6166 check digit for the first 11 characters of an ISIN (Luhn over letter values)."""
    digits = "".join(str(int(ch, 36)) for ch in body.upper())
    total = 0
    for position, ch in enumerate(reversed(digits)):
        value = int(ch) * (2 if position % 2 == 0 else 1)
        total += value // 10 + value % 10
    return str((10 - total % 10) % 10)


def _nsin(domicile: str, rng: random.Random) -> str:
    """9-character national code in the shape each country uses."""
    def pick(alphabet: str, n: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(n))

    if domicile in ("IE", "JE"):
        return "00B" + pick(_SEDOL, 6)
    if domicile == "DE":
        return "000A" + pick(_ALNUM, 5)
    if domicile == "LU":
        return rng.choice("0123") + pick(_DIGITS, 8)
    if domicile == "FR":
        return "001" + pick(_DIGITS, 6)
    if domicile == "ES":
        return "01" + pick(_DIGITS, 7)
    return "00" + pick(_DIGITS, 7)


def _unique_isin(domicile: str, rng: random.Random, used: set[str]) -> str:
    while True:
        body = domicile + _nsin(domicile, rng)
        isin = body + isin_check_digit(body)
        if isin not in used:
            used.add(isin)
            return isin


def _unique_ticker(brand: str, rng: random.Random, used: set[str], reserved: set[str]) -> str:
    """3-5 uppercase letters/digits, usually starting like the brand, never an ordinary word."""
    while True:
        length = rng.choices((3, 4, 5), (0.1, 0.65, 0.25))[0]
        first = brand[:1].upper() if brand[:1].isalpha() and rng.random() < 0.6 else rng.choice(_LETTERS)
        rest = [rng.choice(_LETTERS) for _ in range(length - 1)]
        if rng.random() < 0.25:
            rest[-1] = rng.choice(_DIGITS)
        ticker = first + "".join(rest)
        if ticker not in used and ticker.lower() not in reserved:
            used.add(ticker)
            return ticker


# --- rendering --------------------------------------------------------------------------
_NO_ACCENTS = str.maketrans("áéíóúÁÉÍÓÚñÑü", "aeiouAEIOUnNu")


def _pick_brand(brand: str, rng: random.Random) -> str:
    if rng.random() < P_BRAND_ABSENT:
        return ""
    draw = rng.random()
    for variant, probability in BRAND_VARIANTS.get(brand, ()):
        if draw < probability:
            return variant
        draw -= probability
    return brand


def _num(value: float) -> str:
    """Display form matching the importer: integral floats lose ``.0``."""
    return str(int(value)) if value == int(value) else repr(value)


def generate_records(rows: int, seed: int = 7) -> list[FundRecord]:
    """``rows`` unique funds with their name parts; deterministic for a given seed."""
    if rows < 0:
        raise ValueError("rows must be >= 0")
    rng = random.Random(seed)
    funds = _catalog(rng)
    pools: dict[str, list[tuple[float, int, int]]] = {g: [] for g in GROUP_SHARES}
    for index, fund in enumerate(funds):
        for position, share in enumerate(_share_classes(fund.style, fund.family, fund.letters, fund.currency)):
            key = math.log(rng.random() or 1e-300) / (fund.weight * share.weight)
            pools[_group(fund.kind)].append((key, index, position))
    quotas = _quotas(rows, {group: len(pool) for group, pool in pools.items()})

    records: list[FundRecord] = []
    used_names: set[str] = set()
    sizes: dict[int, float] = {}
    for group, pool in pools.items():
        taken = 0
        for _, index, position in sorted(pool, reverse=True):
            if taken == quotas[group]:
                break
            fund = funds[index]
            share = _share_classes(fund.style, fund.family, fund.letters, fund.currency)[position]
            record = _render(fund, share, position == 0, rng, used_names, sizes, index)
            if record is not None:
                records.append(record)
                taken += 1
    rng.shuffle(records)

    # Tickers must not double as words of the corpus, common query words or brand spellings.
    reserved = set(COMMON_WORDS) | {token for record in records for token in tokens(record.name)}
    reserved.update(t for variants in BRAND_VARIANTS.values() for v, _ in variants for t in tokens(v))
    reserved.update(t for abbreviations in ABBREVIATIONS.values() for a in abbreviations for t in tokens(a))
    isins: set[str] = set()
    tickers: set[str] = set()
    domicile_codes = {name: code for code, name in DOMICILES.items()}
    for record in records:
        record.cells[0] = _unique_isin(domicile_codes[record.cells[9]], rng, isins)
        record.cells[1] = _unique_ticker(record.brand, rng, tickers, reserved)
    return records


def _render(fund: _Fund, share: _Share, main_class: bool, rng: random.Random, used_names: set[str],
            sizes: dict[int, float], index: int) -> FundRecord | None:
    body, label, brand_text = fund.body, share.label, _pick_brand(fund.brand, rng)
    if fund.kind == "FI" and rng.random() < P_NO_ACCENTS:
        body, label, brand_text = (part.translate(_NO_ACCENTS) for part in (body, label, brand_text))
    upper = rng.random() < P_UPPERCASE
    for candidate in (brand_text, fund.brand):
        name = compose_name(candidate, fund.sep, body, label)
        name = name.upper() if upper else name
        if normalize(name) not in used_names:
            brand_text = candidate
            break
    else:
        return None
    used_names.add(normalize(name))

    if index not in sizes:
        sizes[index] = math.exp(rng.gauss(LOG_AUM[fund.kind], 1.0)) * fund.weight ** 0.3
    aum = max(0.1, round(sizes[index] * share.weight ** 0.8 * rng.uniform(0.6, 1.4), 1))
    ter = max(0.03, round(fund.ter * share.ter_factor, 2))
    year = fund.launch if main_class else min(2025, fund.launch + rng.randint(0, 7))
    launch = dt.date(year, rng.randint(1, 12), rng.randint(1, 28))
    asset = CATEGORIES[fund.category][0]
    cells = [
        "", "", name, fund.company, fund.category, asset, fund.region, share.currency,
        "Acc" if share.acc else "Dist", DOMICILES[fund.domicile], _num(ter), _num(aum), launch.isoformat(),
    ]
    return FundRecord(
        cells=cells, company=fund.company, brand=fund.brand, brand_text=brand_text, sep=fund.sep, body=body,
        share_class=label, short_name=fund.short_name, kind=fund.kind, upper=upper,
    )


def generate(rows: int, seed: int = 7) -> tuple[list[str], list[list[str]]]:
    """Columns and display-string rows of a ``rows``-row universe (deterministic per seed)."""
    return list(COLUMNS), [record.cells for record in generate_records(rows, seed)]


# --- xlsx -------------------------------------------------------------------------------
def write_xlsx(path: str | Path, columns: list[str], rows: list[list[str]], sheet: str = "Funds") -> None:
    """Write a formatted workbook: numbers as numbers, dates as real Excel dates."""
    import xlsxwriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"strings_to_numbers": False, "strings_to_formulas": False, "strings_to_urls": False}
    workbook = xlsxwriter.Workbook(str(path), options)
    try:
        worksheet = workbook.add_worksheet(sheet)
        header = workbook.add_format({"bold": True, "bg_color": "#EEF2FF", "bottom": 1})
        formats = {name: workbook.add_format({"num_format": fmt}) for name, fmt in NUMBER_COLUMNS.items()}
        date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
        worksheet.write_row(0, 0, columns, header)
        kinds = ["number" if c in NUMBER_COLUMNS else "date" if c in DATE_COLUMNS else "text" for c in columns]
        for r, row in enumerate(rows, start=1):
            for c, (value, kind) in enumerate(zip(row, kinds)):
                if not value:
                    continue
                if kind == "number":
                    worksheet.write_number(r, c, float(value), formats[columns[c]])
                elif kind == "date":
                    worksheet.write_datetime(r, c, dt.date.fromisoformat(value), date_format)
                else:
                    worksheet.write_string(r, c, value)
        for c, name in enumerate(columns):
            longest = max([len(name)] + [len(row[c]) for row in rows[:500]])
            worksheet.set_column(c, c, min(60, max(9, longest + 2)))
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, len(rows), len(columns) - 1)
    finally:
        workbook.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic fund/ETF workbook.")
    parser.add_argument("--rows", type=int, default=2500, help="number of funds (default 2500)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output .xlsx (default {DEFAULT_OUT})")
    parser.add_argument("--seed", type=int, default=7, help="random seed (default 7)")
    args = parser.parse_args(argv)
    columns, rows = generate(args.rows, args.seed)
    write_xlsx(args.out, columns, rows)
    print(f"Wrote {len(rows):,} rows to {args.out} ({args.out.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

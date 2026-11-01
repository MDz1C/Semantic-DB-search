"""Column statistics and role suggestions (ARCHITECTURE.md section 5, R3)."""

from __future__ import annotations

import itertools
import re
import time
from collections.abc import Callable
from typing import TypeVar

import pytest

from semdb.core.profile import profile_columns, suggest_roles
from semdb.core.types import ColumnProfile, Role, Table

T = TypeVar("T")

IDENTIFIER_REASON = "Unique code per row — exact matches identify a row"  # quoted by the spec

BRANDS = [  # (brand that leads the fund name, company in the Brokerage column, index provider)
    ("iShares", "BlackRock", "MSCI"),
    ("Vanguard", "Vanguard", "FTSE"),
    ("Xtrackers", "DWS", "MSCI"),
    ("Amundi", "Amundi", "MSCI"),
]
REGIONS = ["Japan", "Europe", "World", "USA", "Emerging Markets"]
COLUMNS = ["ISIN", "Fund name", "Brokerage", "Grp", "Region", "Currency", "TER", "Launch", "Description", "Notes"]


def _table(columns: list[str], rows: list[list[str]]) -> Table:
    return Table("funds.csv", "", list(columns), [list(row) for row in rows], list(range(2, len(rows) + 2)))


def _percent(reason: str) -> int:
    found = re.search(r"(\d+) ?%", reason)
    assert found, reason
    return int(found.group(1))


@pytest.fixture(scope="module")
def universe() -> Table:
    """80 funds whose names start with their brand ("iShares MSCI Japan UCITS ETF USD Acc"): the
    region sits inside the name and the currency near its end.  Brokerage holds the company
    (BlackRock), Grp the brand under a header that carries no identity keyword."""
    rows = []
    combos = itertools.product(BRANDS, REGIONS, ["USD", "EUR"], ["Acc", "Dist"])
    for n, ((brand, company, provider), region, currency, share) in enumerate(combos):
        name = f"{brand} {provider} {region} UCITS ETF {currency} {share}"
        description = (
            f"The {name} fund aims to track the performance of an index of large and mid capitalisation "
            f"companies in {region}; income is {'reinvested' if share == 'Acc' else 'paid out'}; managed by {company}."
        )
        ter = "n/a" if n % 10 == 9 else f"0.{n % 9 + 1:02d}"  # a few placeholders, as in real sheets
        launch = f"20{10 + n % 15}-{n % 12 + 1:02d}-{n % 28 + 1:02d}" if n % 4 else f"{n % 28 + 1} Jan 2019"
        rows.append([f"IE00B{n:06d}X", name, company, brand, region, currency, ter, launch, description, ""])
    return _table(COLUMNS, rows)


@pytest.fixture(scope="module")
def profiles(universe: Table) -> dict[str, ColumnProfile]:
    return {profile.name: profile for profile in profile_columns(universe)}


# --- statistics ---------------------------------------------------------------------------


def test_one_profile_per_column_in_order(universe: Table, profiles: dict[str, ColumnProfile]) -> None:
    assert list(profiles) == universe.columns
    assert [profile.index for profile in profiles.values()] == list(range(len(universe.columns)))
    assert (profiles["Fund name"].non_empty, profiles["Fund name"].unique) == (80, 80)
    assert (profiles["Grp"].non_empty, profiles["Grp"].unique) == (80, 4)
    assert (profiles["Notes"].non_empty, profiles["Notes"].unique) == (0, 0)
    assert profiles["Currency"].avg_len == 3.0


def test_samples_and_reasons(universe: Table, profiles: dict[str, ColumnProfile]) -> None:
    for profile in profiles.values():
        assert isinstance(profile.reason, str) and profile.reason.strip(), profile.name
        samples = profile.samples
        assert len(samples) <= 5 and len(set(samples)) == len(samples), profile.name
        assert all(isinstance(sample, str) and sample for sample in samples), profile.name
        values = set(universe.column_values(profile.name))
        assert all(any(value.startswith(sample.rstrip("…")) for value in values) for sample in samples)
        assert bool(samples) == bool(profile.non_empty), profile.name
    assert profiles["Grp"].samples == ["iShares", "Vanguard", "Xtrackers", "Amundi"]  # first seen first
    assert profiles["Currency"].samples == ["USD", "EUR"]


def test_long_samples_stay_distinct() -> None:
    prefix = "This exchange-traded fund seeks to track the investment results of an index composed of equities from "
    rows = [[f"Fund {n}", prefix + region] for n, region in enumerate(REGIONS)]
    samples = profile_columns(_table(["Fund name", "Description"], rows))[1].samples
    assert samples and len(set(samples)) == len(samples), samples


# --- deciding columns -----------------------------------------------------------------------


def test_brokerage_keyword_and_isin_identifier_are_deciding(profiles: dict[str, ColumnProfile]) -> None:
    brokerage, isin = profiles["Brokerage"], profiles["ISIN"]
    assert brokerage.suggested_role is Role.DECIDING
    assert "brokerage" in brokerage.reason.lower()
    assert isin.is_identifier and isin.suggested_role is Role.DECIDING
    assert isin.reason == IDENTIFIER_REASON


@pytest.mark.parametrize(
    "header", ["Gestora", "Fund family", "Issuer", "Brokers", "IssuerName", "Management company", "Marca"]
)
def test_identity_keywords_in_the_header(header: str) -> None:
    rows = [[f"Global Equity Fund {n}", ["Alpha Capital", "Beta Partners", "Gamma Invest"][n % 3]] for n in range(12)]
    profile = profile_columns(_table(["Fund name", header], rows))[1]
    assert profile.suggested_role is Role.DECIDING, profile.reason


def test_unique_codes_are_deciding_without_a_keyword() -> None:
    rows = [[f"Global Equity Fund {n}", f"IE00B{n:06d}X"] for n in range(12)]
    code = profile_columns(_table(["Fund name", "Code"], rows))[1]
    assert code.is_identifier and code.suggested_role is Role.DECIDING
    assert code.reason == IDENTIFIER_REASON


def test_brand_column_leading_the_names_is_deciding(profiles: dict[str, ColumnProfile]) -> None:
    """No keyword in "Grp", but its values open every fund name: an identity hidden in the text (R4)."""
    grp = profiles["Grp"]
    assert grp.suggested_role is Role.DECIDING
    assert "'Fund name'" in grp.reason and _percent(grp.reason) == 100


@pytest.mark.parametrize(("branded", "expected"), [(8, Role.DECIDING), (4, Role.SEARCH)], ids=["40%", "20%"])
def test_contamination_needs_30_percent_of_rows(branded: int, expected: Role) -> None:
    """The brand opens ``branded`` of 20 names: 40 % makes it an identity, 20 % does not."""
    rows = []
    for n in range(20):
        brand = "Alpha" if n % 2 else "Omega"
        rows.append([f"{brand} Global Equity Fund {n}" if n < branded else f"Global Equity Fund {n}", brand])
    grp = profile_columns(_table(["Fund name", "Grp"], rows))[1]
    assert grp.suggested_role is expected, grp.reason
    if expected is Role.DECIDING:
        assert "'Fund name'" in grp.reason and _percent(grp.reason) == 40


def test_values_inside_or_at_the_end_of_names_are_not_identities(
    universe: Table, profiles: dict[str, ColumnProfile]
) -> None:
    """Region ("iShares MSCI Japan ...") and Currency ("... USD Acc") repeat inside every name,
    but an identity leads the name; these are attributes, i.e. descriptive text."""
    names = universe.column_values("Fund name")
    for column in ("Region", "Currency"):
        values = universe.column_values(column)
        assert all(value in name for value, name in zip(values, names))  # as repeated as Grp...
        assert profiles[column].suggested_role is Role.SEARCH, profiles[column].reason  # ...yet not deciding


# --- display only and search ------------------------------------------------------------------


def test_numbers_and_dates_are_display_only(profiles: dict[str, ColumnProfile]) -> None:
    ter, launch = profiles["TER"], profiles["Launch"]
    assert ter.numeric_ratio >= 0.8 and ter.suggested_role is Role.IGNORE
    assert launch.date_ratio >= 0.8 and launch.suggested_role is Role.IGNORE


def test_long_prose_is_display_only(profiles: dict[str, ColumnProfile]) -> None:
    description = profiles["Description"]
    assert description.avg_len > 160
    assert description.suggested_role is Role.IGNORE


def test_empty_column_is_display_only(profiles: dict[str, ColumnProfile]) -> None:
    notes = profiles["Notes"]
    assert notes.suggested_role is Role.IGNORE and notes.samples == []


def test_fund_name_is_search(profiles: dict[str, ColumnProfile]) -> None:
    assert profiles["Fund name"].suggested_role is Role.SEARCH
    assert profiles["Fund name"].reason == "Descriptive text"


def test_suggest_roles_matches_the_profiles(universe: Table, profiles: dict[str, ColumnProfile]) -> None:
    roles = suggest_roles(universe)
    assert roles == {name: profile.suggested_role for name, profile in profiles.items()}
    assert roles == {
        "ISIN": Role.DECIDING, "Fund name": Role.SEARCH, "Brokerage": Role.DECIDING, "Grp": Role.DECIDING,
        "Region": Role.SEARCH, "Currency": Role.SEARCH, "TER": Role.IGNORE, "Launch": Role.IGNORE,
        "Description": Role.IGNORE, "Notes": Role.IGNORE,
    }  # fmt: skip


# --- guardrail: there is always something to search ------------------------------------------


def test_guardrail_makes_the_longest_text_column_search() -> None:
    companies = ["BlackRock", "Vanguard", "State Street"]
    rows = [[f"IE00B{n:06d}X", companies[n % 3], f"0.{n % 9 + 1:02d}"] for n in range(30)]
    table = _table(["ISIN", "Brokerage", "TER"], rows)
    profiles = profile_columns(table)
    assert [profile.suggested_role for profile in profiles] == [Role.DECIDING, Role.SEARCH, Role.IGNORE]
    assert profiles[1].reason  # explains why an identity-looking column is searched
    assert suggest_roles(table) == {"ISIN": Role.DECIDING, "Brokerage": Role.SEARCH, "TER": Role.IGNORE}


def test_suggest_roles_adds_a_search_column_to_any_profiles(funds_table: Table) -> None:
    profiles = profile_columns(funds_table)
    for profile in profiles:  # e.g. profiles edited so that nothing is searched any more
        if profile.suggested_role is Role.SEARCH:
            profile.suggested_role = Role.IGNORE
    roles = suggest_roles(funds_table, profiles)
    assert list(roles) == funds_table.columns
    assert [name for name, role in roles.items() if role is Role.SEARCH] == ["Fund name"]  # longest text


def test_table_without_rows_still_gets_a_search_column() -> None:
    table = _table(["Fund name", "Brokerage"], [])
    profiles = profile_columns(table)
    assert [profile.non_empty for profile in profiles] == [0, 0]
    assert all(profile.reason for profile in profiles)
    assert list(suggest_roles(table).values()).count(Role.SEARCH) == 1


# --- speed ------------------------------------------------------------------------------------


def _big_universe(rows: int = 45_000) -> Table:
    brands = [("iShares", "BlackRock"), ("Vanguard", "Vanguard"), ("Xtrackers", "DWS"), ("Amundi", "Amundi"),
              ("SPDR", "State Street"), ("Invesco", "Invesco")]  # fmt: skip
    indices = ["S&P 500", "MSCI World", "MSCI EM", "FTSE All-World", "Nasdaq 100", "Euro Stoxx 50", "DAX"]
    regions = ["US", "Global", "Emerging", "Japan", "Europe", "Germany"]
    columns = ["ISIN", "Ticker", "Fund name", "Brokerage", "Grp", "Region", "Currency", "Asset class", "TER",
               "AUM", "Launch", "Description", "Notes"]  # fmt: skip
    data = []
    for n in range(rows):
        brand, company = brands[n % len(brands)]
        index, currency = indices[(n // 7) % len(indices)], ["USD", "EUR", "GBP"][n % 3]
        data.append([
            f"IE{n:09d}{'ABCDEFGH'[n % 8]}", f"T{n:05d}", f"{brand} {index} UCITS ETF {currency} Series {n}", company,
            brand, regions[(n // 5) % len(regions)], currency, ["Equity", "Bonds", "Commodity"][n % 3],
            f"0.{n % 90 + 10}", f"{(n * 7919) % 99_991 + 10:,}", f"20{n % 20 + 10}-{n % 12 + 1:02d}-{n % 28 + 1:02d}",
            f"Seeks to track the {index} index; share class {n}.", "",
        ])  # fmt: skip
    return Table("universe.xlsx", "Funds", columns, data, list(range(2, rows + 2)))


def _best_time(run: Callable[[], T], limit: float, attempts: int = 2) -> tuple[float, T]:
    """Fastest of up to ``attempts`` runs (a second run only when the first misses ``limit``,
    so a busy machine does not fail the test)."""
    best, result = float("inf"), None
    for _ in range(attempts):
        start = time.perf_counter()
        result = run()
        best = min(best, time.perf_counter() - start)
        if best < limit:
            break
    return best, result  # type: ignore[return-value]


def test_profiling_45k_rows_by_13_columns_is_fast() -> None:
    table = _big_universe()
    assert (len(table), len(table.columns)) == (45_000, 13)
    elapsed, profiles = _best_time(lambda: profile_columns(table), limit=2.0)
    assert elapsed < 2.0, f"profiling took {elapsed:.2f} s"
    roles = {profile.name: profile.suggested_role for profile in profiles}
    assert roles == {
        "ISIN": Role.DECIDING, "Ticker": Role.DECIDING, "Fund name": Role.SEARCH, "Brokerage": Role.DECIDING,
        "Grp": Role.DECIDING, "Region": Role.SEARCH, "Currency": Role.SEARCH, "Asset class": Role.SEARCH,
        "TER": Role.IGNORE, "AUM": Role.IGNORE, "Launch": Role.IGNORE, "Description": Role.SEARCH, "Notes": Role.IGNORE,
    }  # fmt: skip

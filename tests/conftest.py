"""Shared pytest fixtures. Every test runs with an isolated data folder and offscreen Qt."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from semdb.core.types import Role, Table  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "appdata"
    monkeypatch.setenv("SEMDB_DATA_DIR", str(data))
    return data


@pytest.fixture
def model_dir() -> Path:
    path = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
    if not (path / "model.onnx").exists():
        pytest.skip("embedding model not downloaded (run tools/fetch_model.py)")
    return path


@pytest.fixture
def funds_table() -> Table:
    """A tiny fund universe exercising the iShares/Vanguard confusion from the project brief.

    The brokerage column holds the company (BlackRock) while fund names carry the brand
    (iShares, sometimes abbreviated), exactly the contamination the family model handles.
    """
    columns = ["ISIN", "Fund name", "Brokerage", "Currency", "TER"]
    data = [
        ["IE00B5BMR087", "iShares Core S&P 500 UCITS ETF USD (Acc)", "BlackRock", "USD", "0.07"],
        ["IE0031442068", "iShares Core S&P 500 UCITS ETF USD (Dist)", "BlackRock", "USD", "0.07"],
        ["IE00B4L5Y983", "iShares Core MSCI World UCITS ETF USD (Acc)", "BlackRock", "USD", "0.20"],
        ["IE00BKM4GZ66", "iShares Core MSCI EM IMI UCITS ETF", "BlackRock", "USD", "0.18"],
        ["IE00B3XXRP09", "Vanguard S&P 500 UCITS ETF USD Distributing", "Vanguard", "USD", "0.07"],
        ["IE00BFMXXD54", "Vanguard S&P 500 UCITS ETF USD Accumulating", "Vanguard", "USD", "0.07"],
        ["IE00BK5BQT80", "Vanguard FTSE All-World UCITS ETF USD Accumulating", "Vanguard", "USD", "0.22"],
        ["IE00B3RBWM25", "Vanguard FTSE All-World UCITS ETF USD Distributing", "Vanguard", "USD", "0.22"],
        ["LU1681048804", "Amundi S&P 500 UCITS ETF EUR (C)", "Amundi", "EUR", "0.15"],
        ["LU1437016972", "Amundi MSCI Emerging Markets UCITS ETF", "Amundi", "EUR", "0.20"],
        ["IE00BJ0KDQ92", "Xtrackers MSCI World UCITS ETF 1C", "DWS", "USD", "0.19"],
        ["LU0274211480", "Xtrackers DAX UCITS ETF 1C", "DWS", "EUR", "0.09"],
        ["IE00B6YX5C33", "SPDR S&P 500 UCITS ETF (Dist)", "State Street", "USD", "0.03"],
        ["IE00BFY0GT14", "SPDR MSCI World UCITS ETF", "State Street", "USD", "0.12"],
        ["IE00BYX2JD69", "iShares MSCI World SRI UCITS ETF EUR (Acc)", "BlackRock", "EUR", "0.20"],
        ["IE00B52VJ196", "iShares MSCI Europe SRI UCITS ETF EUR (Acc)", "BlackRock", "EUR", "0.20"],
    ]
    return Table(
        source_name="funds.xlsx",
        sheet_name="Funds",
        columns=columns,
        rows=[list(r) for r in data],
        source_rows=list(range(2, len(data) + 2)),
    )


@pytest.fixture
def funds_roles() -> dict[str, Role]:
    return {
        "ISIN": Role.DECIDING,
        "Fund name": Role.SEARCH,
        "Brokerage": Role.DECIDING,
        "Currency": Role.IGNORE,
        "TER": Role.IGNORE,
    }

"""Column statistics and automatic role suggestions (deciding / search / display only).

See docs/ARCHITECTURE.md, section 5.  Suggestions only pre-fill the import dialog; the user
has the final word.  Two safety refinements on top of the spec's rules:

* digit-only "codes" of varying width (amounts, counts) are numbers, not identifiers, and
  a numeric column is never suggested as deciding: a fund size of 500 must not turn the
  "500" in "S&P 500" into an identity;
* an identity keyword in the header ("Company name") does not make a column deciding when
  it holds a different value in almost every row: then it names the item itself.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .table import is_date_text, is_number_text
from .text import is_identifier_column, looks_like_code, tokens
from .types import ColumnProfile, Role, Roles, Table

IDENTITY_KEYWORDS = (
    "broker", "brokerage", "gestora", "issuer", "emisor", "emisora", "provider", "proveedor",
    "family", "families", "familia", "manager", "management company", "sociedad gestora",
    "promoter", "promotora", "sponsor", "company", "companies", "compania", "empresa", "brand",
    "marca", "isin", "ticker", "symbol", "simbolo", "cusip", "sedol", "wkn",
)  # fmt: skip
CODE_KEYWORDS = frozenset({"isin", "ticker", "symbol", "simbolo", "cusip", "sedol", "wkn"})

NUMERIC_SHARE = 0.8  # numeric/date ratio from which a column is display-only
LOW_CARDINALITY = 0.2  # unique / filled at or below which a column can be a "family"
CONTAMINATION_SHARE = 0.3  # share of rows whose text repeats the column's value
LEAD_TOKENS = 2  # a repeated identity leads the text it contaminates (brand before product name)
LONG_TEXT = 160  # average length above which text is prose (display only)
MOSTLY_EMPTY = 0.95

_PLACEHOLDERS = frozenset({"-", "--", "—", "–", "n/a", "na", "n.a.", "n.a", "nan", "null", "none", "#n/a", "?"})
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_SAMPLE_CHARS = 100


@dataclass(slots=True)
class _Stats:
    index: int
    name: str
    values: tuple[str, ...]
    counts: Counter[str]  # non-empty value -> occurrences, in first-seen order
    non_empty: int
    avg_len: float
    numeric_ratio: float
    date_ratio: float
    is_identifier: bool

    @property
    def unique(self) -> int:
        return len(self.counts)

    @property
    def is_text(self) -> bool:
        return self.non_empty > 0 and self.numeric_ratio < NUMERIC_SHARE and self.date_ratio < NUMERIC_SHARE


def profile_columns(table: Table) -> list[ColumnProfile]:
    """One profile per column, in column order, each with a suggested role and reason."""
    n_rows = len(table.rows)
    columns = list(zip(*table.rows)) if table.rows else [() for _ in table.columns]
    stats = [_column_stats(index, name, values) for index, (name, values) in enumerate(zip(table.columns, columns))]
    contamination = _contamination(stats)
    profiles: list[ColumnProfile] = []
    for column in stats:
        role, reason = _suggest(column, n_rows, contamination.get(column.index))
        profiles.append(
            ColumnProfile(
                name=column.name,
                index=column.index,
                non_empty=column.non_empty,
                unique=column.unique,
                avg_len=column.avg_len,
                numeric_ratio=column.numeric_ratio,
                date_ratio=column.date_ratio,
                is_identifier=column.is_identifier,
                suggested_role=role,
                reason=reason,
                samples=_samples(column.counts),
            )
        )
    fallback = _search_fallback(profiles)
    if fallback is not None:
        fallback.suggested_role = Role.SEARCH
        fallback.reason = "Used for search: the longest text column (no other descriptive text found)"
    return profiles


def suggest_roles(table: Table, profiles: list[ColumnProfile] | None = None) -> Roles:
    """Role for every column, from the profiles' suggestions."""
    if profiles is None:
        profiles = profile_columns(table)
    suggested = {profile.name: profile.suggested_role for profile in profiles}
    roles = {name: suggested.get(name, Role.IGNORE) for name in table.columns}
    if roles and Role.SEARCH not in roles.values():
        fallback = _search_fallback([profile for profile in profiles if profile.name in roles])
        roles[fallback.name if fallback is not None else table.columns[0]] = Role.SEARCH
    return roles


# --------------------------------------------------------------------------------------


def _column_stats(index: int, name: str, values: tuple[str, ...]) -> _Stats:
    counts = Counter(values)
    non_empty = len(values) - counts.pop("", 0)
    avg_len = numeric_ratio = date_ratio = 0.0
    if non_empty:
        avg_len = sum(len(value) * count for value, count in counts.items()) / non_empty
        measured = non_empty - sum(count for value, count in counts.items() if value.casefold() in _PLACEHOLDERS)
        if measured:
            numeric_ratio = sum(count for value, count in counts.items() if is_number_text(value)) / measured
            date_ratio = sum(count for value, count in counts.items() if is_date_text(value)) / measured
    return _Stats(
        index=index,
        name=name,
        values=values,
        counts=counts,
        non_empty=non_empty,
        avg_len=avg_len,
        numeric_ratio=numeric_ratio,
        date_ratio=date_ratio,
        is_identifier=_is_identifier(values, counts, non_empty),
    )


def _is_identifier(values: tuple[str, ...], counts: Counter[str], non_empty: int) -> bool:
    """``text.is_identifier_column``, skipped early when cheap counts already rule it out."""
    if non_empty < 2 or len(counts) < 0.9 * non_empty:
        return False
    if sum(count for value, count in counts.items() if looks_like_code(value)) < 0.9 * non_empty:
        return False
    return is_identifier_column(list(values))


def _suggest(column: _Stats, n_rows: int, contamination: tuple[str, float] | None) -> tuple[Role, str]:
    if column.non_empty == 0:
        return Role.IGNORE, "Empty column"
    if column.is_identifier and not _is_measure(column):
        return Role.DECIDING, "Unique code per row — exact matches identify a row"
    keyword = _identity_keyword(column.name)
    if keyword and column.is_text and (keyword in CODE_KEYWORDS or not _names_items(column)):
        return Role.DECIDING, f"Identity field (the header mentions '{keyword}')"
    if contamination is not None:
        other, share = contamination
        return Role.DECIDING, f"Its values also appear inside '{other}' in {round(share * 100)} % of rows"
    if column.non_empty <= (1 - MOSTLY_EMPTY) * n_rows:
        return Role.IGNORE, "Almost always empty"
    if column.unique <= 1:
        return Role.IGNORE, "The same value in every row"
    if column.numeric_ratio >= NUMERIC_SHARE:
        return Role.IGNORE, "Numbers — shown with results, not searched"
    if column.date_ratio >= NUMERIC_SHARE:
        return Role.IGNORE, "Dates — shown with results, not searched"
    if column.avg_len > LONG_TEXT:
        return Role.IGNORE, "Long free text — shown with results, not searched"
    return Role.SEARCH, "Descriptive text"


def _is_measure(column: _Stats) -> bool:
    """Distinct digit-only values of varying width are amounts or counts, not codes."""
    digits = [value for value in column.counts if value.isdigit()]
    if len(digits) < 0.9 * column.unique:
        return False
    widths = Counter(len(value) for value in digits)
    return widths.most_common(1)[0][1] < 0.9 * len(digits)


def _names_items(column: _Stats) -> bool:
    """A (nearly) different value in every row: the column names the item, not its family."""
    return column.non_empty >= 50 and column.unique >= 0.9 * column.non_empty


def _identity_keyword(header: str) -> str | None:
    """Most specific identity keyword in the header: a phrase, a word, then a word prefix
    ("Brokers", "IssuerID" after CamelCase splitting)."""
    words = tokens(_CAMEL_RE.sub(" ", header))
    phrase = f" {' '.join(words)} "
    singles = [keyword for keyword in IDENTITY_KEYWORDS if " " not in keyword]
    return (
        next((k for k in IDENTITY_KEYWORDS if " " in k and f" {k} " in phrase), None)
        or next((k for k in singles if k in words), None)
        or next((k for k in singles if len(k) >= 4 and any(word.startswith(k) for word in words)), None)
    )


def _contamination(stats: list[_Stats]) -> dict[int, tuple[str, float]]:
    """Low-cardinality text columns whose values the row's descriptive text repeats.

    A value token counts in a row when it is one of the first words of the row's text (brands
    lead the names they repeat in: "iShares Core ...", "Vanguard FTSE ..."; a region in "MSCI
    Japan" or a currency at the end is an attribute, not an identity) and the token is precise
    in that text: at least half of the rows whose text contains it belong to values that
    contain it (so "Fund" in "XYZ Fund Managers" does not count, "Vanguard" does).
    Returns ``{column index: (descriptive column name, share of filled rows)}``.
    """
    candidates = [
        column
        for column in stats
        if column.is_text and not column.is_identifier and 2 <= column.unique <= LOW_CARDINALITY * column.non_empty
    ]
    targets = [
        column
        for column in stats
        if column.is_text
        and not column.is_identifier
        and column.unique > LOW_CARDINALITY * column.non_empty
        and column.avg_len <= LONG_TEXT
    ]
    found: dict[int, tuple[str, float]] = {}
    if not candidates or not targets:
        return found
    row_tokens: dict[int, list[tuple[frozenset[str], frozenset[str]]]] = {}
    for column in candidates:
        value_tokens = {value: _identity_tokens(value) for value in column.counts}
        vocabulary = frozenset().union(*value_tokens.values())
        if not vocabulary:
            continue
        best: tuple[str, float] | None = None
        for target in targets:
            if target.unique <= column.unique:
                continue
            if target.index not in row_tokens:
                row_tokens[target.index] = _tokenize_rows(target.values)
            share, hits = _repeat_share(column, value_tokens, vocabulary, row_tokens[target.index])
            if hits >= 3 and share >= CONTAMINATION_SHARE and (best is None or share > best[1]):
                best = (target.name, share)
        if best is not None:
            found[column.index] = best
    return found


def _identity_tokens(value: str) -> frozenset[str]:
    words = tokens(value)
    single = len(set(words)) == 1
    return frozenset(word for word in words if not word.isdigit() and (len(word) >= 3 or (single and len(word) == 2)))


def _tokenize_rows(values: tuple[str, ...]) -> list[tuple[frozenset[str], frozenset[str]]]:
    """Per row: (all tokens, the first LEAD_TOKENS tokens longer than one character)."""
    cache: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    out: list[tuple[frozenset[str], frozenset[str]]] = []
    for value in values:
        entry = cache.get(value)
        if entry is None:
            toks = tokens(value)
            lead = [token for token in toks if len(token) > 1][:LEAD_TOKENS]
            entry = cache[value] = (frozenset(toks), frozenset(lead))
        out.append(entry)
    return out


def _repeat_share(
    column: _Stats,
    value_tokens: dict[str, frozenset[str]],
    vocabulary: frozenset[str],
    row_tokens: list[tuple[frozenset[str], frozenset[str]]],
) -> tuple[float, int]:
    seen: Counter[str] = Counter()  # rows whose text contains the token
    joint: Counter[str] = Counter()  # ... and whose own value contains it too
    repeats: list[frozenset[str]] = []  # per such row: own-value tokens leading its text
    for value, (toks, lead) in zip(column.values, row_tokens):
        present = toks & vocabulary
        if not present:
            continue
        seen.update(present)
        own = value_tokens.get(value)
        if own:
            mine = present & own
            if mine:
                joint.update(mine)
                repeats.append(mine & lead)
    precise = {token for token, count in joint.items() if count >= 0.5 * seen[token]}
    hits = sum(1 for mine in repeats if not mine.isdisjoint(precise))
    return hits / column.non_empty, hits


def _search_fallback(profiles: list[ColumnProfile]) -> ColumnProfile | None:
    """The guardrail: when nothing is SEARCH, the longest text column that is not an identifier."""
    if not profiles or any(profile.suggested_role is Role.SEARCH for profile in profiles):
        return None
    pools = (
        [p for p in profiles if p.non_empty and not p.is_identifier and p.numeric_ratio < NUMERIC_SHARE and p.date_ratio < NUMERIC_SHARE],
        [p for p in profiles if p.non_empty and not p.is_identifier],
        [p for p in profiles if p.non_empty],
        profiles,
    )  # fmt: skip
    pool = next(pool for pool in pools if pool)
    return max(pool, key=lambda profile: (profile.avg_len, -profile.index))


def _samples(counts: Counter[str], limit: int = 5) -> list[str]:
    """Up to ``limit`` distinct values *as displayed* (shortened values can collide)."""
    shown: dict[str, None] = {}
    for value in counts:
        shown.setdefault(_shorten(value), None)
        if len(shown) == limit:
            break
    return list(shown)


def _shorten(value: str) -> str:
    return value if len(value) <= _SAMPLE_CHARS else value[: _SAMPLE_CHARS - 1] + "…"

"""Text folding, tokenization and span utilities shared by every search component.

All matching happens on *folded* text: Unicode NFKD with combining marks removed, then
casefolded, so "Gestión" and "GESTION" compare equal.  A token is a run of letters or
digits; an '&' or apostrophe joining two runs stays inside the token, so "S&P" and
"L'Oreal" are single tokens.  Span helpers report positions in the *original* string,
which lets callers cut identity fragments out of a query while keeping its casing.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W_]+(?:[&'][^\W_]+)*")
_ALNUM_SPLIT_RE = re.compile(r"[^\W\d_]+(?:[&'][^\W\d_]+)*|\d+")
_SPACES_RE = re.compile(r"\s+")
_EMPTY_BRACKETS_RE = re.compile(r"[(\[{]\s*[)\]}]")
_SEPARATORS = " \t-–—|·•:,;/\\"
_DOUBLE_SEPARATOR_RE = re.compile(r"\s+[-–—|·•:,;/]\s+(?:[-–—|·•:,;/]\s+)+")


def fold(text: str) -> str:
    """Accent-strip and casefold ``text`` (keeps punctuation and length semantics loose)."""
    if text.isascii():
        return text.lower()
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold().replace("’", "'").replace("‘", "'")


def tokens(text: str) -> list[str]:
    """Folded tokens of ``text``: ``tokens("iShares Core S&P 500")`` -> ``['ishares', 'core', 's&p', '500']``."""
    return _TOKEN_RE.findall(fold(text))


def normalize(text: str) -> str:
    """Folded tokens joined by single spaces: the canonical comparable form of a string."""
    return " ".join(tokens(text))


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """``(folded_token, start, end)`` for each token, with offsets into the original ``text``."""
    if text.isascii():
        return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text.lower())]
    pieces: list[str] = []
    owners: list[int] = []
    for index, char in enumerate(text):
        folded = fold(char)
        pieces.append(folded)
        owners.extend([index] * len(folded))
    folded_text = "".join(pieces)
    return [
        (m.group(), owners[m.start()], owners[m.end() - 1] + 1)
        for m in _TOKEN_RE.finditer(folded_text)
    ]


def split_alnum(token: str) -> list[str]:
    """Split a short mixed letter/digit token at letter-digit boundaries.

    ``"s&p500" -> ["s&p", "500"]``, ``"3yr" -> ["3", "yr"]``.  Tokens longer than 8
    characters (ISINs and other codes) and tokens that are all letters or all digits
    are returned unchanged as a one-element list.
    """
    if len(token) > 8:
        return [token]
    has_digit = any(ch.isdigit() for ch in token)
    has_alpha = any(ch.isalpha() for ch in token)
    if not (has_digit and has_alpha):
        return [token]
    parts = _ALNUM_SPLIT_RE.findall(token)
    return parts if len(parts) > 1 else [token]


def ngrams(toks: list[str], max_n: int = 3) -> list[tuple[str, int, int]]:
    """Contiguous token n-grams as ``(phrase, first_token, last_token_exclusive)``, n = 1..max_n."""
    out: list[tuple[str, int, int]] = []
    for n in range(1, max_n + 1):
        for start in range(0, len(toks) - n + 1):
            out.append((" ".join(toks[start : start + n]), start, start + n))
    return out


def tidy(text: str) -> str:
    """Collapse whitespace and drop separators/brackets left dangling after removals."""
    text = _EMPTY_BRACKETS_RE.sub(" ", text)
    text = _DOUBLE_SEPARATOR_RE.sub(" - ", text)
    text = _SPACES_RE.sub(" ", text)
    return text.strip(_SEPARATORS)


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Cut ``[start, end)`` character spans out of ``text`` and tidy the result."""
    if not spans:
        return tidy(text)
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append(" ")
        cursor = end
    pieces.append(text[cursor:])
    return tidy("".join(pieces))


def looks_like_code(raw: str) -> bool:
    """True for identifier-looking values: ISINs, CUSIPs, tickers, numeric ids.

    Codes are a single word (no spaces) and either mix letters with digits, are short
    all-caps words (tickers such as ``VOO``), or are long digit strings.
    """
    value = raw.strip()
    if not (2 <= len(value) <= 24) or any(ch.isspace() for ch in value):
        return False
    has_digit = any(ch.isdigit() for ch in value)
    has_alpha = any(ch.isalpha() for ch in value)
    if has_digit and has_alpha:
        return True
    if has_alpha and not has_digit:
        return value.isupper() and len(value) <= 6 and value.isalnum()
    return has_digit and len(value) >= 5 and value.isdigit()


def is_identifier_column(values: list[str]) -> bool:
    """True when a column holds per-row codes: >= 90 % of non-empty values look like codes
    and >= 90 % of them are distinct (ISIN, ticker, internal id)."""
    filled = [value for value in values if value.strip()]
    if len(filled) < 2:
        return False
    code_like = sum(1 for value in filled if looks_like_code(value))
    distinct = len({normalize(value) for value in filled})
    return code_like >= 0.9 * len(filled) and distinct >= 0.9 * len(filled)

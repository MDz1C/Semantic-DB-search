"""Reading spreadsheet files into :class:`Table` objects (offline, no macros, no formulas run).

Excel-family formats are parsed with python-calamine (Rust, memory-safe, never executes
macros); CSV/TSV/TXT with the standard library.  Every cell becomes a display string.

Raw rows (``read_raw``, ``detect_header_row`` and the ``header_row`` argument of
``read_table``) are aligned with the sheet: raw row ``i`` is sheet row ``i + 1``, so a
1-based header-row number shown in the UI is the row number the user sees in Excel.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
import struct
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from python_calamine import (
    CalamineError,
    CalamineWorkbook,
    PasswordError,
    SheetTypeEnum,
    SheetVisibleEnum,
)

from . import isolation
from .text import fold
from .types import CancelFn, Cancelled, ProgressFn, Table

EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xlsb", ".xls", ".ods")
TEXT_EXTENSIONS = (".csv", ".tsv", ".txt")
SUPPORTED_EXTENSIONS = EXCEL_EXTENSIONS + TEXT_EXTENSIONS

READING_STAGE = "Reading rows…"
HEADER_SCAN_ROWS = 30  # header candidates are searched among the first rows only
MAX_CELLS = 10_000_000  # rows x columns of a sheet; far above the ~45k-row target

_HEADER_WINDOW = HEADER_SCAN_ROWS + 10  # rows looked at by detect_header_row
_CANCEL_EVERY = 2_000
_DELIMITERS = ",;\t|"
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
_SNIFF_CHARS = 65_536
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_CFB_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
_ENCRYPTED_MARKER = "EncryptedPackage".encode("utf-16-le")  # stream name inside an encrypted .xlsx
_END_OF_CHAIN = 0xFFFFFFFE  # OLE2 sector chain terminator
_BIFF_EOF, _BIFF_BOUNDSHEET, _BIFF_DIMENSIONS = 0x000A, 0x0085, 0x0200
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")

_BREAKS_RE = re.compile(r"\s*[\r\n\t\v\f\x85\u2028\u2029]+\s*")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:[,.'\u00a0\u202f ]\d{3})+|\d+)(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?"
)
_NUMBER_DECORATION = "$€£¥₹%‰ \u00a0"
_MONTH_WORDS = (
    "january february march april may june july august september october november december "
    "enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre "
    "diciembre janvier fevrier mars avril mai juin juillet aout septembre octobre novembre "
    "decembre januar februar marz maerz juni juli oktober dezember "
    "jan feb mar apr jun jul aug sep sept oct nov dec ene abr ago dic janv fevr avr juil okt mrz dez"
)
_MONTH = "(?:" + "|".join(sorted(set(_MONTH_WORDS.split()), key=len, reverse=True)) + r")\.?"
_TIME = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[ap]\.?m\.?)?"
_DATE_RE = re.compile(
    rf"""
    (?:
        \d{{4}}[/.-]\d{{1,2}}[/.-]\d{{1,2}}
      | \d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}
      | \d{{1,2}}(?:st|nd|rd|th)?[ /.-]?{_MONTH}[ /.,-]*\d{{2,4}}
      | {_MONTH}[ /.-]?\d{{1,2}}(?:st|nd|rd|th)?,?[ /.-]\d{{2,4}}
      | {_MONTH}[ /.,-]*\d{{4}}
    )
    (?:[ t]{_TIME}(?:z|[+-]\d{{2}}:?\d{{2}})?)?
    | {_TIME}
    """,
    re.VERBOSE,
)


class TableError(Exception):
    """User-facing import problem (unsupported type, password-protected, empty sheet...)."""


# --------------------------------------------------------------------------------------
# public API


def list_sheets(path: str | Path) -> list[str]:
    """Visible sheet names in workbook order (``[""]`` for text files)."""
    if isolation.in_process():
        return _list_sheets_here(Path(path))
    return _isolated("list_sheets", Path(path), {})


def read_raw(path: str | Path, sheet: str | None = None, max_rows: int | None = None) -> list[list[str]]:
    """First ``max_rows`` raw rows of a sheet as display strings (no header handling), for previews.

    Rows are padded to a common width; leading empty columns of a worksheet are left out.
    """
    if isolation.in_process():
        return _read_raw_here(Path(path), sheet, max_rows)
    return _isolated("read_raw", Path(path), {"sheet": sheet, "max_rows": max_rows})


def read_table(
    path: str | Path,
    sheet: str | None = None,
    header_row: int | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> Table:
    """Read a whole sheet. ``header_row`` None means auto-detect. Raises TableError.

    Unless running inside the import worker, the file is parsed in a separate memory-capped
    process (see :mod:`semdb.core.isolation`), so a malicious file cannot exhaust memory.
    """
    if isolation.in_process():
        return _read_table_here(Path(path), sheet, header_row, progress, cancel)
    data = _isolated("read_table", Path(path), {"sheet": sheet, "header_row": header_row}, progress, cancel)
    table = Table(data["source_name"], data["sheet_name"], data["columns"], data["rows"], data["source_rows"])
    if progress is not None:
        progress(READING_STAGE, 1.0)
    return table


def _isolated(op: str, source: Path, payload: dict, progress: ProgressFn | None = None,
              cancel: CancelFn | None = None):
    """Run a reader in the import worker, translating its failures into TableError."""
    _source_kind(source)  # instant answers (missing file, wrong type) without a worker
    try:
        return isolation.run(op, {"path": str(source.resolve()), **payload}, progress, cancel)
    except isolation.WorkerError as exc:
        if exc.kind == "table":
            raise TableError(exc.message) from None
        if exc.kind == "memory":
            raise TableError(
                f"'{source.name}' was not opened: reading it needs more memory than is safe "
                "(it declares an enormous sheet). The file may be damaged or malicious. If you "
                "trust it, open it in Excel and save a copy with just the data you need."
            ) from None
        if exc.kind == "timeout":
            raise TableError(f"Reading '{source.name}' took too long and was stopped.") from None
        raise TableError(
            f"'{source.name}' could not be read: the file reader stopped unexpectedly. The file "
            "may be damaged or malicious. Try opening it in Excel and saving it again."
        ) from None


def _list_sheets_here(source: Path) -> list[str]:
    if _source_kind(source) == "text":
        return [""]
    with _open_workbook(source) as workbook:
        return _sheet_names(workbook, source)


def _read_raw_here(source: Path, sheet: str | None, max_rows: int | None) -> list[list[str]]:
    if _source_kind(source) == "text":
        rows = _read_text_rows(source, max_rows=max_rows)
    else:
        rows, _, _ = _read_sheet_rows(source, sheet, max_rows=max_rows)
    width = max(map(len, rows), default=0)
    return [row + [""] * (width - len(row)) if len(row) < width else row for row in rows]


def detect_header_row(raw_rows: list[list[str]]) -> int:
    """0-based index of the most plausible header row within the first rows of a sheet.

    A header has >= 2 filled cells (and at least half as many as the widest nearby row),
    mostly non-numeric text, and the next non-empty row has a similar or larger number of
    filled cells.  When no row qualifies the text condition is dropped (headers made of
    years), and failing that row 0 is used.  Only the first ``HEADER_SCAN_ROWS + 10``
    rows are inspected, so a preview of that many rows gives the same answer as the file.
    """
    window = raw_rows[:_HEADER_WINDOW]
    counts = [sum(1 for cell in row if cell) for row in window]
    if not counts:
        return 0
    widest = max(counts)

    def next_count(index: int) -> int:
        return next((count for count in counts[index + 1 :] if count), 0)

    def qualifies(index: int, need_text: bool) -> bool:
        count = counts[index]
        if count < 2 or count < 0.5 * widest:
            return False
        if need_text:
            texty = sum(1 for cell in window[index] if cell and not _is_numeric_like(cell))
            if texty * 2 <= count:
                return False
        return next_count(index) >= 0.5 * count

    candidates = range(min(HEADER_SCAN_ROWS, len(window)))
    for need_text in (True, False):
        for index in candidates:
            if qualifies(index, need_text):
                return index
    return 0


def _read_table_here(
    source: Path,
    sheet: str | None,
    header_row: int | None,
    progress: ProgressFn | None,
    cancel: CancelFn | None,
) -> Table:
    def tick(fraction: float) -> None:
        if cancel is not None and cancel():
            raise Cancelled()
        if progress is not None:
            progress(READING_STAGE, min(max(fraction, 0.0), 1.0))

    tick(0.0)
    if _source_kind(source) == "text":
        rows = _read_text_rows(source, tick=tick)
        sheet_name, column_offset = "", 0
    else:
        rows, sheet_name, column_offset = _read_sheet_rows(source, sheet, tick=tick)
    table = _build_table(rows, header_row, source.name, sheet_name, column_offset)
    tick(1.0)
    return table


# --------------------------------------------------------------------------------------
# cell values -> display strings


def cell_text(value: object) -> str:
    """Display string of one calamine cell value (see ARCHITECTURE.md, section 4)."""
    kind = type(value)
    if kind is str:
        return _clean_text(value)  # type: ignore[arg-type]
    if kind is float:
        return _format_float(value)  # type: ignore[arg-type]
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, dt.datetime):
        return _format_datetime(value)
    if isinstance(value, dt.date):
        return f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
    if isinstance(value, dt.time):
        return _format_clock(value.hour * 3600 + value.minute * 60 + value.second, value.microsecond)
    if isinstance(value, dt.timedelta):
        return _format_duration(value)
    return _clean_text(str(value))


def _clean_text(value: str) -> str:
    text = value.strip()
    if text.isprintable():
        return text
    text = _BREAKS_RE.sub(" ", text)
    return _CONTROL_RE.sub("", text).strip()


def _format_float(value: float) -> str:
    if value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    text = f"{value:.15g}"  # Excel shows at most 15 significant digits: 0.1 + 0.2 -> 0.3
    if "e" in text and 1e-10 <= abs(value) < 1e21:
        text = format(Decimal(text), "f")
    return text


def _format_datetime(value: dt.datetime) -> str:
    if value.microsecond >= 500_000 and value.year < 9999:
        value += dt.timedelta(seconds=1)  # Excel serials carry float noise: 13:44:59.9999 is 13:45
    date_text = f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
    seconds = value.hour * 3600 + value.minute * 60 + value.second
    return f"{date_text} {_format_clock(seconds, 0)}" if seconds else date_text


def _format_clock(seconds: int, microsecond: int) -> str:
    seconds = (seconds + (microsecond >= 500_000)) % 86_400
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if secs else f"{hours:02d}:{minutes:02d}"


def _format_duration(value: dt.timedelta) -> str:
    total = round(value.total_seconds())
    sign = "-" if total < 0 else ""
    hours, rest = divmod(abs(total), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{sign}{hours}:{minutes:02d}:{secs:02d}" if secs else f"{sign}{hours}:{minutes:02d}"


# --------------------------------------------------------------------------------------
# value classification (shared with core.profile)


def is_number_text(value: str) -> bool:
    """True for numbers as people type them: ``1,234.5``, ``0,07``, ``-3``, ``12 %``, ``(1.5)``, ``$100``."""
    text = value.strip().strip(_NUMBER_DECORATION)
    if text[:1] == "(" and text[-1:] == ")":
        text = text[1:-1].strip(_NUMBER_DECORATION)
    return bool(text) and _NUMBER_RE.fullmatch(text) is not None


def is_date_text(value: str) -> bool:
    """True for common date, date-time and time spellings (ISO, D/M/Y, ``31 Jan 2024``, ``13:45``)."""
    text = value.strip()
    if not text or not text[0].isalnum():
        return False
    return _DATE_RE.fullmatch(fold(text)) is not None


def _is_numeric_like(value: str) -> bool:
    return is_number_text(value) or is_date_text(value)


# --------------------------------------------------------------------------------------
# file detection and workbooks


def _source_kind(source: Path) -> str:
    """'workbook' or 'text', judged by the extension and the file's first bytes."""
    extension = source.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise TableError(
            f"'{source.name}' is not a supported file type. Choose an Excel workbook "
            "(.xlsx, .xlsm, .xlsb, .xls, .ods) or a text file (.csv, .tsv, .txt)."
        )
    if not source.is_file():
        raise TableError(f"'{source.name}' was not found. It may have been moved or deleted.")
    try:
        with source.open("rb") as handle:
            head = handle.read(4096)
    except OSError as exc:
        raise TableError(f"'{source.name}' could not be opened ({exc.strerror or exc}).") from exc
    if not head.strip():
        raise TableError(f"'{source.name}' is empty.")
    if head.startswith((*_ZIP_MAGICS, _CFB_MAGIC)):
        return "workbook"  # also catches a workbook renamed to .csv
    if not _looks_like_text(head):
        if extension in TEXT_EXTENSIONS:
            raise TableError(f"'{source.name}' does not look like a text file. Is it the right file?")
        return "workbook"  # let calamine try (and explain) other binary formats
    if extension in EXCEL_EXTENSIONS and head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"<"):
        raise TableError(
            f"'{source.name}' is a web page or XML file saved with an Excel extension. "
            "Open it in Excel, save it as .xlsx, then import that file."
        )
    return "text"  # includes 'Excel' exports that are really CSV/TSV text


def _looks_like_text(head: bytes) -> bool:
    if head.startswith(_UTF16_BOMS):
        return True
    return not any(byte < 9 or (13 < byte < 32 and byte not in (26, 27)) for byte in head)


@contextmanager
def _open_workbook(source: Path) -> Iterator[CalamineWorkbook]:
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise TableError(f"'{source.name}' could not be opened ({exc.strerror or exc}).") from exc
    damaged = (
        f"'{source.name}' could not be read. The file may be damaged, or it is not a real "
        "Excel/OpenDocument workbook. Try opening it in Excel and saving it again."
    )
    if data.startswith(_CFB_MAGIC):
        streams = _ole_streams(data)
        if streams is None:
            # Unparseable container: its directory may still name an encrypted package.
            raise TableError(_password_message(source) if _ENCRYPTED_MARKER in data else damaged)
        if "encryptedpackage" in streams:  # an encrypted .xlsx is an OLE container
            raise TableError(_password_message(source))
        book = streams.get("workbook", streams.get("book"))
        if book is not None and not _biff_dimensions_sound(book):
            raise TableError(damaged)
    try:
        with _calamine_errors(damaged):
            # Content-based format detection: also opens workbooks whose extension is wrong.
            workbook = CalamineWorkbook.from_filelike(io.BytesIO(data))
    except TableError as exc:
        if isinstance(exc.__cause__, PasswordError):
            raise TableError(_password_message(source)) from exc.__cause__
        raise
    try:
        yield workbook
    finally:
        workbook.close()


@contextmanager
def _calamine_errors(message: str) -> Iterator[None]:
    """Report calamine failures, including Rust panics, as a TableError with ``message``."""
    try:
        yield
    except CalamineError as exc:
        raise TableError(message) from exc
    except BaseException as exc:  # pyo3 raises Rust panics as PanicException(BaseException)
        if type(exc).__name__ != "PanicException":
            raise
        raise TableError(message) from exc


def _password_message(source: Path) -> str:
    return (
        f"'{source.name}' is password-protected. Open it in Excel, remove the password "
        "(File > Info > Protect Workbook > Encrypt with Password), save it, then import it again."
    )


def _ole_streams(data: bytes) -> dict[str, bytes] | None:
    """Streams of an OLE2 compound file (.xls container) by case-folded name, or None when
    the container is unsound.

    calamine trusts the container's sector numbers and sizes: on a damaged .xls it can panic
    or request gigabytes of memory and abort the whole process.  So the header, FAT, DIFAT,
    directory and every stream chain are checked against the file length first (MS-CFB).
    """
    if len(data) < 1024:
        return None
    major, byte_order, shift, mini_shift = struct.unpack_from("<4H", data, 0x1A)
    if byte_order != 0xFFFE or (major, shift) not in ((3, 9), (4, 12)) or mini_shift != 6:
        return None
    size = 1 << shift
    _, n_fat, first_dir, _, cutoff, first_mini, n_mini, first_difat, n_difat = struct.unpack_from("<9I", data, 0x28)
    count = -(-(len(data) - size) // size)  # sectors after the header; the last may be short
    if cutoff != 4096 or not 0 < n_fat <= count or n_mini > count or n_difat > count:
        return None

    def sector(index: int) -> bytes:
        return data[(index + 1) * size : (index + 2) * size].ljust(size, b"\0")

    def chain(start: int, table: Sequence[int], limit: int) -> list[int] | None:
        sectors: list[int] = []
        seen: set[int] = set()
        while start != _END_OF_CHAIN:
            if start >= limit or start >= len(table) or start in seen:
                return None
            seen.add(start)
            sectors.append(start)
            start = table[start]
        return sectors

    per_sector = size // 4
    fat_sectors = list(struct.unpack_from("<109I", data, 0x4C))
    difat = first_difat
    for _ in range(n_difat):
        if difat >= count:
            return None
        values = struct.unpack(f"<{per_sector}I", sector(difat))
        fat_sectors.extend(values[:-1])
        difat = values[-1]
    fat_sectors = fat_sectors[:n_fat]
    if len(fat_sectors) < n_fat or any(index >= count for index in fat_sectors):
        return None
    fat = struct.unpack(f"<{n_fat * per_sector}I", b"".join(sector(index) for index in fat_sectors))

    directory = chain(first_dir, fat, count)
    if not directory:
        return None
    raw = b"".join(sector(index) for index in directory)
    entries: list[tuple[int, str, int, int]] = []  # (type, name, first sector, stream size)
    for offset in range(0, len(raw), 128):
        kind = raw[offset + 0x42]
        if kind in (1, 2, 5):
            name_bytes = min(struct.unpack_from("<H", raw, offset + 0x40)[0], 64)
            name = raw[offset : offset + name_bytes].decode("utf-16-le", "replace").rstrip("\0")
            start = struct.unpack_from("<I", raw, offset + 0x74)[0]
            length = struct.unpack_from("<Q" if major == 4 else "<I", raw, offset + 0x78)[0]
            entries.append((kind, name, start, length))
    if not entries or entries[0][0] != 5:
        return None

    _, _, root_start, root_size = entries[0]
    root = chain(root_start, fat, count) if root_size else []
    mini = chain(first_mini, fat, count) if n_mini else []
    if root is None or len(root) * size < root_size or mini is None or len(mini) < n_mini:
        return None
    mini_fat = struct.unpack(f"<{len(mini) * per_sector}I", b"".join(sector(index) for index in mini))
    mini_stream = b"".join(sector(index) for index in root)[:root_size]
    streams: dict[str, bytes] = {}
    for kind, name, start, length in entries[1:]:
        if kind != 2:
            continue
        if length == 0:
            streams[name.casefold()] = b""
        elif length < cutoff:  # small streams live in 64-byte sectors of the mini stream
            sectors = chain(start, mini_fat, -(-root_size // 64))
            if sectors is None or len(sectors) * 64 < length:
                return None
            streams[name.casefold()] = b"".join(mini_stream[i * 64 : i * 64 + 64] for i in sectors)[:length]
        else:
            sectors = chain(start, fat, count)
            if sectors is None or len(sectors) * size < length:
                return None
            streams[name.casefold()] = b"".join(sector(index) for index in sectors)[:length]
    return streams


def _biff_dimensions_sound(stream: bytes) -> bool:
    """True when every DIMENSIONS record stays within the BIFF limits (65,536 x 256).

    calamine reserves memory for rows x columns from this record, so a corrupted one would
    make it request petabytes.  Records are walked the way calamine reads them: the globals
    up to EOF, then each sheet from the offset its BOUNDSHEET record gives.
    """
    sheet_starts: list[int] = []

    def walk(position: int) -> bool:
        while position + 4 <= len(stream):
            kind, length = struct.unpack_from("<HH", stream, position)
            body = position + 4
            if kind == _BIFF_EOF:
                return True
            if kind == _BIFF_BOUNDSHEET and length >= 4 and not sheet_starts_done:
                sheet_starts.append(struct.unpack_from("<I", stream, body)[0])
            elif kind == _BIFF_DIMENSIONS:
                if length >= 14:
                    first_row, last_row, first_col, last_col = struct.unpack_from("<IIHH", stream, body)
                elif length >= 8:
                    first_row, last_row, first_col, last_col = struct.unpack_from("<HHHH", stream, body)
                else:
                    return False
                if not (first_row <= last_row <= 65_536 and first_col <= last_col <= 256):
                    return False
            position = body + length
        return True

    sheet_starts_done = False
    if not walk(0):
        return False
    sheet_starts_done = True
    return all(start < len(stream) and walk(start) for start in sheet_starts)


def _sheet_names(workbook: CalamineWorkbook, source: Path) -> list[str]:
    worksheets = [meta for meta in workbook.sheets_metadata if meta.typ == SheetTypeEnum.WorkSheet]
    visible = [meta.name for meta in worksheets if meta.visible == SheetVisibleEnum.Visible]
    names = visible or [meta.name for meta in worksheets] or list(workbook.sheet_names)
    if not names:
        raise TableError(f"'{source.name}' contains no worksheets.")
    return names


def _read_sheet_rows(
    source: Path,
    sheet: str | None,
    max_rows: int | None = None,
    tick: Callable[[float], None] | None = None,
) -> tuple[list[list[str]], str, int]:
    """(rows aligned with sheet rows, sheet name, index of the first used column)."""
    with _open_workbook(source) as workbook:
        names = _sheet_names(workbook, source)
        if not sheet:
            name = names[0]
        elif sheet in workbook.sheet_names:
            name = sheet
        else:
            raise TableError(f"There is no sheet named '{sheet}' in '{source.name}'.")
        with _calamine_errors(f"Sheet '{name}' in '{source.name}' could not be read. The file may be damaged."):
            data = workbook.get_sheet_by_name(name)
            start = data.start
            if start is None:
                return [], name, 0
            top, left = start
            width = data.width
            if data.height * width > MAX_CELLS:
                raise TableError(
                    f"Sheet '{name}' is too large to import ({data.height:,} rows x {width:,} "
                    "columns). Delete unused rows and columns in Excel, save, and try again."
                )
            limit = None if max_rows is None else max(max_rows - top, 0)
            values = data.to_python(skip_empty_area=True, nrows=limit) if limit != 0 else []

    lead = top if max_rows is None else min(top, max_rows)
    rows: list[list[str]] = [[""] * width for _ in range(lead)]
    total = len(values) or 1
    convert = cell_text
    for index, row in enumerate(values):
        if tick is not None and index % _CANCEL_EVERY == 0:
            tick(0.9 * index / total)
        rows.append([convert(value) for value in row])
    return rows, name, left


# --------------------------------------------------------------------------------------
# delimited text


def _read_text_rows(
    source: Path, max_rows: int | None = None, tick: Callable[[float], None] | None = None
) -> list[list[str]]:
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise TableError(f"'{source.name}' could not be opened ({exc.strerror or exc}).") from exc
    text = _decode(data).replace("\x00", "")
    if source.suffix.lower() == ".tsv":
        delimiter = "\t"
    else:
        sample = text[:_SNIFF_CHARS]
        if len(text) > _SNIFF_CHARS and "\n" in sample:
            sample = sample[: sample.rindex("\n")]
        delimiter = _detect_delimiter(sample)

    csv.field_size_limit(max(csv.field_size_limit(), 1 << 24))
    buffer = io.StringIO(text, newline="")
    reader = csv.reader(buffer, delimiter=delimiter)
    total = len(text) or 1
    rows: list[list[str]] = []
    clean = _clean_text
    try:
        for index, record in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            if tick is not None and index % _CANCEL_EVERY == 0:
                tick(0.9 * buffer.tell() / total)
            rows.append([clean(field) for field in record])
    except csv.Error as exc:
        raise TableError(
            f"'{source.name}' could not be read as delimited text (line {reader.line_num}: {exc})."
        ) from exc
    return rows


def _decode(data: bytes) -> str:
    if data.startswith(_UTF16_BOMS):
        return data.decode("utf-16", errors="replace")
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodes any byte string")


def _detect_delimiter(sample: str) -> str:
    """The delimiter among ``,;<tab>|`` that splits the sample's lines most consistently.

    Ties (same consistency and column count) go to ``csv.Sniffer``'s choice; a candidate
    that yields one column everywhere never wins, and a file without any delimiter reads
    as one column.
    """
    scores: dict[str, tuple[float, int]] = {}
    for delimiter in _DELIMITERS:
        try:
            widths = [
                len(record)
                for record in csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter)
                if any(field.strip() for field in record)
            ]
        except csv.Error:
            widths = []
        if not widths:
            scores[delimiter] = (0.0, 1)
            continue
        mode, frequency = Counter(widths).most_common(1)[0]
        scores[delimiter] = (frequency / len(widths) if mode >= 2 else 0.0, mode)

    best = max(_DELIMITERS, key=lambda d: (scores[d], -_DELIMITERS.index(d)))
    if scores[best][0] == 0.0:
        return ","
    try:
        sniffed = csv.Sniffer().sniff(sample[:8192], delimiters=_DELIMITERS).delimiter
    except csv.Error:
        return best
    return sniffed if scores.get(sniffed) == scores[best] else best


# --------------------------------------------------------------------------------------
# header + rows -> Table


def _build_table(
    rows: list[list[str]], header_row: int | None, source_name: str, sheet_name: str, column_offset: int
) -> Table:
    where = f"sheet '{sheet_name}' of '{source_name}'" if sheet_name else f"'{source_name}'"
    if not any(any(row) for row in rows):
        raise TableError(f"There is no data in {where}.")
    if header_row is None:
        header_row = detect_header_row(rows)
    if not 0 <= header_row < len(rows):
        raise TableError(f"Row {header_row + 1} is outside {where}, which has {len(rows):,} rows.")

    width = max(map(len, rows))
    header = rows[header_row] + [""] * (width - len(rows[header_row]))
    data = rows[header_row + 1 :]

    filled = [bool(label) for label in header]
    for row in data:
        for index, cell in enumerate(row):
            if cell:
                filled[index] = True
    keep = [index for index in range(width) if filled[index]]
    labels = [header[index] or f"Column {column_offset + index + 1}" for index in keep]
    columns = _unique_names(labels)

    table_rows: list[list[str]] = []
    source_rows: list[int] = []
    for number, row in enumerate(data, start=header_row + 2):
        if len(row) >= width:
            cells = [row[index] for index in keep]
        else:
            cells = [row[index] if index < len(row) else "" for index in keep]
        if any(cells):
            table_rows.append(cells)
            source_rows.append(number)
    if not table_rows:
        raise TableError(f"There are no data rows below the header (row {header_row + 1}) in {where}.")
    return Table(
        source_name=source_name,
        sheet_name=sheet_name,
        columns=columns,
        rows=table_rows,
        source_rows=source_rows,
    )


def _unique_names(labels: list[str]) -> list[str]:
    """Keep the first occurrence of each label; later duplicates become 'Label (2)', 'Label (3)'..."""
    taken = set(labels)
    used: set[str] = set()
    names: list[str] = []
    for label in labels:
        name = label
        if name in used:
            suffix = 2
            while f"{label} ({suffix})" in taken or f"{label} ({suffix})" in used:
                suffix += 1
            name = f"{label} ({suffix})"
        used.add(name)
        names.append(name)
    return names

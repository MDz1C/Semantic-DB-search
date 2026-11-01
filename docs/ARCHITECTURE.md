# Semantic DB Search — Architecture & Specification

This document is the single source of truth for how the app works. Code must match it;
if an implementation needs to deviate, the deviation is documented here.

## 1. What the product must do (requirements)

Derived from the original project brief (a long design conversation, kept out of git as
`Project_scope.txt`). Requirement ids are referenced from code reviews and tests.

| Id | Requirement |
|----|-------------|
| R1 | Desktop app installed on Windows like any program (Setup.exe, Start-menu entry, uninstaller). **No web server, no browser, no open network port, no network access at runtime.** |
| R2 | Import a spreadsheet database (Excel `.xlsx/.xlsm/.xlsb/.xls`, `.ods`, `.csv/.tsv/.txt`). Target size ~45,000 rows; must stay fast. |
| R3 | After import the user ticks **deciding columns**: identity fields (brokerage, issuer, family, ISIN, ticker…) that must not be mixed up. Example: a query for an *iShares* fund must not return a *Vanguard* fund first just because the rest of the words overlap better. |
| R4 | Deciding information often also appears inside descriptive text (the brand inside the fund name), possibly abbreviated (`ISHS`, `iSh`) or worded differently from the deciding column (`iShares` in names vs `BlackRock` in the brokerage column). This must be handled **automatically, with no manual alias table**: learn aliases from the data, remove high-confidence repeats from the text that gets compared, keep uncertain text, and let structured reranking decide. |
| R5 | Search is "intelligent": meaning-aware (abbreviations, synonyms, misspellings, Spanish↔English), but also exact where it matters (`S&P 500` ≠ `S&P 400`, ISIN/ticker exact hits). |
| R6 | Short queries (just a name, or name + brand) work: "this looks like an iShares fund → look at iShares funds → compare the names". |
| R7 | **Results per query** control at the **top left**. **Import** (with the usual import icon) and **Reset** buttons at the **top right**. |
| R8 | A big paste box at the **bottom**: each line is one query (pasting a column from Excel works). |
| R9 | The middle shows the output: one **collapsible list per query** (query = header, ranked similar rows inside), all **collapsed** at first, with a button to **collapse all** at once. |
| R10 | The list of queries can be ordered by **recency** or **alphabetically**. |
| R11 | **Reset** wipes the output. Output **persists after closing** the app. |
| R12 | Beautiful, simple, intuitive interface; light and dark themes. |
| R13 | Nice to have from the brief, all implemented: export results to Excel (query with its matches side by side, like the brief's "spreadsheet formula" idea), saved deciding-column choice, ranked-result explanations ("why this match"). |

## 2. Shape of the solution

```
                ┌──────────────── UI (PySide6, main thread) ────────────────┐
 file ──► table.read_table ──► profile.suggest_roles ──► Import dialog (user ticks roles)
                │                                                             │
                ▼                                                             ▼
          storage.Store  ◄──── worker threads ────►  engine.SearchIndex.build
          (SQLite: dataset,                          ├─ families.FamilyModel   (identity)
           history, settings,                        ├─ lexical.LexicalIndex   (exact/typo)
           embedding cache)                          └─ embedder.OnnxEmbedder  (meaning)
                ▲                                                             │
                └──── HistoryEntry ◄── SearchIndex.search_many(queries) ◄─────┘
```

* `src/semdb/core/` — pure Python + numpy, no Qt. Testable headless.
* `src/semdb/storage.py` — SQLite persistence.
* `src/semdb/ui/` — Qt widgets. Long work runs on worker threads; widgets are only touched on the main thread.
* `src/semdb/selftest.py` — headless end-to-end check used on packaged builds (`--selftest`).
* `tools/` — model download/quantization, sample data, evaluation, icon generation (dev-time only).
* `packaging/` — PyInstaller spec, Inno Setup installer script, one-shot `build.ps1`.

Shared types live in `core/types.py`; shared text utilities in `core/text.py` (already implemented —
use them, do not re-implement folding/tokenizing). Public signatures in the module stubs are fixed.

## 3. Text conventions (`core/text.py`)

* `fold` = NFKD, strip combining marks, casefold. `tokens` = folded runs of letters/digits (`&` and `'`
  inside a token are kept: `s&p`, `l'oreal`). `normalize` = tokens joined by one space.
* `token_spans` gives original-string offsets so fragments can be cut from a query keeping its casing;
  `remove_spans` cuts and tidies dangling separators/brackets.
* `split_alnum` splits short mixed tokens (`s&p500` → `s&p`,`500`; `3yr` → `3`,`yr`); codes > 8 chars stay whole.
* `looks_like_code` flags identifier-looking values (ISIN, CUSIP, tickers, numeric ids).

## 4. Importing (`core/table.py`)

* Excel-family files via **python-calamine** (`CalamineWorkbook.from_path`); text files via `csv` with
  `csv.Sniffer` (delimiter among `,;\t|`), encodings tried in order `utf-8-sig`, `cp1252`, `latin-1`.
* Only visible worksheets are listed. Password-protected or corrupt files raise `TableError` with a
  friendly message. Never evaluate formulas or macros (calamine returns cached values).
* Cell → display string: `None`/empty → `''`; `bool` → `TRUE/FALSE`; `int` → `str`; `float` that is
  integral → no `.0` (`500.0` → `500`), otherwise `repr`-shortest form without exponent for normal magnitudes
  (`0.07` → `0.07`); `datetime` → `YYYY-MM-DD` (plus ` HH:MM` when time ≠ midnight); `date` → ISO;
  `time` → `HH:MM[:SS]`; strings stripped of surrounding whitespace, internal newlines → space.
* `detect_header_row`: among the first 30 raw rows choose the first row that (a) has ≥ 2 non-empty
  cells, (b) whose non-empty cells are mostly non-numeric text, and (c) is followed by a row with a
  similar or larger number of non-empty cells. Fallback 0.
* Headers: empty header → `Column N` (1-based letter-free index); duplicates get ` (2)`, ` (3)`.
  Columns that are empty in the header *and* all data rows are dropped. Fully empty rows are skipped.
  Rows are padded/truncated to the header width. `source_rows` records the 1-based sheet row number.
* Progress reported as `("Reading rows…", fraction)`; honour `cancel` every ~2,000 rows.

## 5. Column roles (`core/profile.py`)

`profile_columns` computes per column: non-empty count, unique count, average length, numeric ratio,
date ratio, `is_identifier` (`text.is_identifier_column`: ≥ 90 % of non-empty values pass
`looks_like_code` **and** ≥ 90 % distinct), up to 5 distinct sample values, and a suggested role with
a one-sentence reason:

1. **Deciding** when any of:
   * header keyword match (folded): `broker, brokerage, gestora, issuer, emisor, emisora, provider,
     proveedor, family, familia, manager, management company, sociedad gestora, promoter, promotora,
     sponsor, company, compania, empresa, brand, marca, isin, ticker, symbol, simbolo, cusip, sedol, wkn`;
   * `is_identifier` (reason: "Unique code per row — exact matches identify a row");
   * **contamination**: a low-cardinality text column (unique ratio ≤ 0.2, ≥ 2 distinct values) whose
     values' distinctive tokens appear inside another text column of the same row in ≥ 30 % of rows
     (reason: "Its values also appear inside 'Fund name' in 64 % of rows").
2. **Display only** when numeric ratio ≥ 0.8, date ratio ≥ 0.8, average length > 160 (long prose),
   unique count ≤ 1, or the column is ≥ 95 % empty.
3. **Search** otherwise (text that describes the item; reason: "Descriptive text").

Guardrail: if no column ends up as Search, the longest-average-length text column that is not an
identifier becomes Search. `suggest_roles` returns `{name: role}` for every column.

## 6. Family model (`core/families.py`) — the identity layer (R3, R4, R6)

Built from the table, the deciding columns and the search columns. Let *text tokens* of a row be the
set of `tokens()` of its search cells, and `df_text[t]` the number of rows whose text tokens contain `t`.

### 6.1 Vocabulary
For each deciding column: distinct values by `normalize()`; display value = first-seen original;
`codes[row]` = value id or −1 for empty cells. `is_identifier` = `text.is_identifier_column(values)`.

### 6.2 Indicators (what in a query points to a value)
`indicators[key] -> [(value_id, weight)]`, keys are normalized tokens or space-joined phrases.

*Identifier columns*: key = normalized value (and its space-less form). Weight 1.0 if the value has a
digit or is ≥ 5 chars, otherwise 1.0 when `df_text[key] == 0` else 0.3 (tickers like `ALL` that are
also ordinary words must not hijack the ranking). No fuzzy matching.

*Other deciding columns*:
* Value tokens. For token `t` of value `v`: `nv[t]` = number of distinct values containing `t`;
  `generic = 1 if nv[t] <= 3 else 3 / nv[t]`; tokens of length ≤ 2 get weight 0 unless the value is a
  single token. Text precision with smoothing α = 2: `prec = (rows in S whose text contains t + α) /
  (df_text[t] + α)` where S is the set of rows whose value contains `t` (so `vanguard` shared by
  "Vanguard Group Ireland" and "Vanguard Investments UK" is still precise). `weight = generic * prec`.
* Full value phrase (≥ 2 tokens) and its space-less join (`i shares` → `ishares`): weight
  `max(0.9 * prec_phrase, best token weight)`, where `prec_phrase` is the same smoothed precision
  computed on phrase occurrences.
* **Learned aliases** (R4, automatic): for columns with ≤ 5,000 distinct values and median family size
  ≥ 3, every text token `t` (length ≥ 3, not purely numeric, not already a value token) with support
  `n(v,t) ≥ 3`, coverage `n(v,t) / |v| ≥ 0.2` and precision `n(v,t) / df_text[t] ≥ 0.85` becomes an
  indicator for `v` with `weight = precision * (0.5 + 0.5 * min(1, coverage / 0.5))`. Example: rows
  with Brokerage = BlackRock whose names start with "iShares" teach `ishares → BlackRock`; `lyxor` can
  map to Amundi. Exposed via `learned_aliases` for the UI.
* When the same key maps to the same value twice keep the max weight.

### 6.3 Query detection
1. `token_spans(query)`; candidate n-grams n = 1..3 plus space-less joins of bigrams.
2. Exact: every n-gram found in `indicators` contributes `(value, weight, span)`.
3. Fuzzy (non-identifier columns only), for query unigrams of length ≥ 3 without an exact hit **and**
   `df_text[token] <= 2` (an established word of the corpus is not a misspelled brand):
   * typo: `rapidfuzz.fuzz.ratio ≥ 85` against unigram keys of length ≥ 4 → `weight * (ratio/100)²`;
   * abbreviation: same first letter, the query token is a character subsequence of the key, and
     `len(token) / len(key) ≥ 0.35` → `weight * 0.6` (`ishs`→`ishares`, `jpm`→`jpmorgan`).
4. Per value: combine contributions with noisy-or `1 − Π(1 − wᵢ)`; drop values below 0.25.
5. `spans` = spans of contributions with weight ≥ 0.5 for values that survived (these fragments are
   removed from the query before text matching). `found` = one `FamilyEvidence` per surviving value,
   strongest first, `matched_text` = original query substring.

### 6.4 Row cleaning (`clean_cells`)
For each search cell of a row: remove (via `remove_spans`) occurrences of indicator keys of the row's
**own** values with weight ≥ 0.5 (exact token/phrase matches only — never fuzzy on rows), plus
contiguous occurrences of the full value phrase. If a cell would become empty, keep it unchanged.
"iShares Core S&P 500 UCITS ETF" with Brokerage = BlackRock → "Core S&P 500 UCITS ETF".

### 6.5 Family factor
For each deciding column with evidence `E` (value → strength), `top = max(E)`. For a row with value
`v`: `conflict = top − E.get(v, 0)`; for an empty cell `conflict = 0.5 * top`. Column factor
`1 − strictness * conflict`; total factor = product over columns (vectorized with `codes`). A row that
belongs to the named family keeps factor 1; a Vanguard row for an iShares query gets ≈ 0.16.
Columns without evidence contribute 1.

## 7. Lexical index (`core/lexical.py`)

Numpy-only inverted index. Features per document (computed on `normalize`d text):
* character 3-grams of each token padded with spaces (`" s&p "` → `" s&"`, `"s&p"`, `"&p "`); tokens of
  length ≤ 1 contribute the padded token itself;
* word features `w:` + token and `w:` + each part of `split_alnum(token)`, with a ×1.5 weight multiplier.
TF = `1 + log(tf)`; IDF = `log((1 + N) / (1 + df)) + 1`; documents L2-normalized. Query vectors use the
same weighting; features unseen in the corpus still count in the query norm with the maximum IDF.
Scores via postings (CSR arrays `indptr`, `doc_ids` int32, `weights` float32) gathered for the query's
features and summed with `np.bincount(..., minlength=N)`. Must build 45k docs in ≤ 5 s and score a
query in ≤ 15 ms. Empty query → zeros.

## 8. Embeddings (`core/embedder.py`, `tools/fetch_model.py`)

* Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, 50+ languages incl.
  Spanish/English), pinned revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
* `tools/fetch_model.py` downloads the fp32 `onnx/model.onnx` and `tokenizer.json` (plus config files)
  at the pinned revision, **verifies SHA-256**, quantizes with
  `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8, per_channel=True, reduce_range=True)`
  (7-bit weights: no u8s8 saturation on AVX2-only CPUs; measured fastest here: ~880 sentences/s,
  cosine fidelity 0.99 vs fp32), checks fidelity on probe sentences, and writes
  `models/paraphrase-multilingual-MiniLM-L12-v2/{model.onnx, tokenizer.json, manifest.json}`.
  Pinned SHA-256: `onnx/model.onnx` `10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a`,
  `tokenizer.json` `2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8`.
* Runtime: `tokenizers.Tokenizer.from_file`, truncation 128, padding per batch; inputs `input_ids`,
  `attention_mask`, `token_type_ids` (zeros) as int64; mean pooling over the attention mask; L2 norm.
  Sort texts by length before batching (64) and restore order. ONNX session: `ORT_ENABLE_ALL`,
  CPU provider, default threads. `model_id` = `"<model name>@<revision[:7]>/q8rr"`. Empty strings embed
  fine (they produce a valid vector). Cancel checked between batches.
* `load_default_embedder()` returns None (and logs why) when the model folder is missing/corrupt; the
  app then runs lexical-only and says so in the status bar.

## 9. Engine (`core/engine.py`)

### 9.1 Build
1. Deciding = columns with role DECIDING, search = role SEARCH (table order). If there is no search
   column, use every non-IGNORE column as search text **without** cleaning. `primary_column` = first
   search column (else first column).
2. `FamilyModel.build` (skipped when `config.use_family` is False).
3. Row text = cleaned search cells joined with `" · "` (empty cells skipped).
4. `LexicalIndex(row_texts)`.
5. Embeddings: deduplicate row texts, look them up in the cache, embed the missing ones (progress
   `"Understanding rows…"`), write them back, then build an `(N, dim)` float32 matrix.
6. Exact-id lookup: for every DECIDING or SEARCH column where `text.is_identifier_column` holds, map
   normalized value (and space-less form) → row indices.
Progress stages: `"Analyzing columns…"`, `"Learning identity aliases…"`, `"Indexing words…"`,
`"Understanding rows…"` (embedding, the long one, with done/total), `"Finalizing…"`.

### 9.2 Ranking a query
1. `detection = family.detect(query)`; `cleaned = remove_spans(query, detection.spans)`.
   If `cleaned` has no tokens, the query only named an identity: `text_score = 0.5` for every row
   (ranking then comes from identity, ties broken by lexical score of the raw query).
2. `lexical = lex.scores(cleaned)`.
3. `semantic = clip((E @ embed(cleaned) − floor) / (1 − floor), 0, 1)` with `floor = config.semantic_floor`.
4. `text_score = w * semantic + (1 − w) * lexical` (`w = config.semantic_weight`; lexical only when
   no embedder).
5. Exact identifier hit (any query token or the whole query, normalized/space-less, found in the id
   lookup) → that row's `text_score = 1` and `exact_id = column`.
6. `factor = family.family_factor(detection, config.strictness)`; `score = text_score * factor`.
7. Top-k with `np.argpartition`, sorted by score desc then row index. `agreements`/`conflicts` list
   the deciding columns where the row's value has full / reduced strength vs the strongest evidence.
`search_many` embeds all cleaned queries in one batch first. 45k rows: ≤ 40 ms per query.

### 9.3 History snapshot
`to_history_entry` copies all cells of each match (aligned with `table.columns`), `source_row`,
rank, score and `detail` (`semantic, lexical, text_score, family_factor, exact_id, agreements,
conflicts`), plus the query evidence, deciding columns, primary column and `table.label`.

## 10. Storage (`storage.py`)

`app.sqlite` (WAL, `check_same_thread=False`, one `threading.Lock` around every operation):
* `settings(key TEXT PRIMARY KEY, value TEXT)` — JSON values.
* `dataset(id INTEGER PRIMARY KEY CHECK (id = 1), source_name, sheet_name, columns_json, roles_json,
  fingerprint, saved_at)` and `dataset_rows(idx INTEGER PRIMARY KEY, source_row INTEGER, cells_json TEXT)`.
  `save_dataset` replaces both in one transaction.
* `history(id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, query_key TEXT, created_at TEXT,
  payload_json TEXT)`; `query_key = normalize(query)`. `add_entries` replaces any existing entry with
  the same `query_key` (re-searching a query refreshes it) and returns entries with ids.
  `load_entries` returns oldest → newest.
`embeddings.sqlite`: `vectors(key BLOB PRIMARY KEY, vec BLOB, used INTEGER)`; key = sha1 of
`model_id + "\x00" + text`; float16 storage, returned as float32; LRU pruning to `max_entries` by `used`.
Schema version in `PRAGMA user_version`; unknown future versions are opened read-only-safe (recreate
cache tables, never crash). Corrupt database files are renamed `*.corrupt-<timestamp>` and recreated.

## 11. Export (`core/export.py`)

xlsxwriter with `{"strings_to_formulas": False, "strings_to_urls": False, "strings_to_numbers": False}`
(formula-injection safe). Sheet **Summary**: `Query`, then for i in 1..k: `Match i` (primary column
value) and `Score i` (percentage format). Sheet **Details**: `Query`, `Rank`, `Match %`, `Sheet row`,
then every column of the entry. Bold header, frozen header row, autofilter, sensible widths.

## 12. User interface (`ui/`)

### 12.1 Visual system
* Qt **Fusion** style + palette + one generated QSS from tokens. Follow the OS light/dark scheme
  (`QStyleHints.colorScheme`, live updates) unless the user picks Light/Dark in the ⋯ menu.
* Font families `["Segoe UI Variable Text", "Segoe UI", "Inter", "Arial"]`, base 10 pt; titles 12–13 pt
  semibold; captions 9 pt. 8 px spacing grid. Radius: cards 12 px, controls 8 px, pills 999 px.
  Control height 34 px.
* Tokens (light / dark):
  `bg #F5F6FA / #0E1014`, `surface #FFFFFF / #161A21`, `surface_alt #F8F9FC / #1B2029`,
  `border #E3E6ED / #262C36`, `border_strong #CDD2DC / #343B47`, `text #111827 / #E7EAF0`,
  `text_muted #6B7280 / #9AA3B2`, `text_faint #9CA3AF / #6B7383`, `accent #4F46E5 / #818CF8`,
  `accent_hover #4338CA / #A5B4FC`, `accent_soft #EEF2FF / #1E2040`, `on_accent #FFFFFF / #0E1014`,
  `success #059669 / #34D399`, `success_soft #ECFDF5 / #0F2A22`, `warning #D97706 / #FBBF24`,
  `warning_soft #FFFBEB / #2B2412`, `danger #DC2626 / #F87171`, `danger_soft #FEF2F2 / #2C1618`,
  `selection #E0E7FF / #272A4D`.
* Icons: inline Lucide-style SVG strings (ISC licence, attribution in `ui/icons.py`), recoloured per
  theme, rendered with QtSvg.

### 12.2 Layout (R7–R11)
```
┌ Header ─────────────────────────────────────────────────────────────────────────────┐
│ ◆  Results per query  [ − 5 + ]                          [⇪ Import]  [↺ Reset]  [⋯] │
├ Dataset bar (only when a database is loaded) ─────────────────────────────────────────┤
│ ▦ funds.xlsx · Funds   45,210 rows   ● Ready   Deciding: (Brokerage) (ISIN)  [Columns…] │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Results 12   [Recent | A–Z]   [⇕ Collapse all]            [⌕ Filter queries]  [Export] │
│ ┌ results tree (card) ─────────────────────────────┐ ┌ Details (toggleable) ─────────┐ │
│ │ ▸ iShares Core S&P 500   BlackRock  5 · 2 min  98% │ │ Vanguard FTSE All-World …     │ │
│ │ ▾ Vanguard FTSE All-World           5 · 1 h    96% │ │ 96 % match                    │ │
│ │    1  96%  Vanguard FTSE All-World UCITS…  Vanguard │ │ ✓ Brokerage agrees            │ │
│ │    2  81%  …                                        │ │ Text 93 % · Meaning 95 % …   │ │
│ └───────────────────────────────────────────────────┘ │ Fund name   …  (all fields)    │ │
│                                                        └───────────────────────────────┘ │
├ splitter ─────────────────────────────────────────────────────────────────────────────┤
│ ┌ Paste queries — one per line (Ctrl+Enter) ─────────────────────────┐  [  Search  ]  │
│ └─────────────────────────────────────────────────────────────────────┘  3 queries     │
├ status bar: message …                                   🔒 Offline · data stays on this PC │
```
* **Results per query**: spin box 1–50 at the top left (R7). Changing it instantly re-renders every
  list (50 results are stored per query; the view shows the first K). Persisted.
* **Import** (top right, primary-outline button with import icon) and **Reset** (ghost button);
  **⋯** menu: Theme (System/Light/Dark), Open data folder, About (version, offline statement,
  licences). Ctrl+O imports.
* **Results tree** (`QTreeView` + custom `QAbstractItemModel` + painting delegate, uniform row heights
  off): top-level rows are queries (first column spanned, 44 px): chevron, query text (semibold),
  detected-identity pills (e.g. `BlackRock`), muted "5 results · 2 min ago", best score pill on the
  right. Children are the first K results (34 px): rank, score pill (green ≥ 85 %, amber ≥ 60 %, grey
  below; thin bar), then the dataset columns ordered primary → other search → deciding → rest.
  Deciding cells render as soft pills with ✓ (agrees) / ✕ (conflicts) when the query named that
  column. Everything starts **collapsed** (R9); a single-query search auto-expands its new list.
  **Collapse all / Expand all** toggle button (Ctrl+Shift+C). Sort segmented control
  **Recent | A–Z** (R10, persisted). Filter box narrows queries by text. Header columns resizable;
  horizontal scrolling. Copy: Ctrl+C copies selected result rows as TSV with header (pastes into
  Excel). Context menu — query: Expand/Collapse, Re-run, Copy query, Copy results, Delete;
  result: Copy row, Copy cell, Copy <primary value>. Delete key removes selected queries (confirm).
* **Details panel**: shows the selected result (title, score breakdown with small bars: Match,
  Text, Meaning, Exact words; identity agreements/conflicts; every field as selectable label/value
  pairs; Copy row) or the selected query (query, what was detected: "Brokerage → BlackRock (from
  'iShares', 99 %)", cleaned query, time, dataset, Re-run / Delete). Toggle button in toolbar; persisted.
* **Composer** (bottom, R8): `QPlainTextEdit` in a card, placeholder "Paste queries here — one per
  line. Each line becomes its own result list.", live count "3 queries", primary **Search** button,
  Ctrl+Enter. Disabled with an explanatory tooltip when there is no database or the index is being
  built. During a search the button becomes **Cancel** with progress. Duplicated lines in a batch are
  searched once; re-searching an existing query refreshes its list. The box clears after success.
* **Reset** (R11): confirmation dialog "Clear all results? N result lists will be removed from this
  computer. Your database stays loaded." → clears history and the composer.
* **Empty states** inside the results card: (1) no database: large icon, "Import your database",
  "Choose an Excel or CSV file. Everything stays on this computer — nothing is uploaded.", primary
  "Import file…", link "Try with sample data", hint "or drop a file anywhere in this window";
  (2) database, no results: "Ready to search 45,210 rows", "Paste one query per line below…", and up to
  3 example chips taken from the data that fill the composer when clicked; (3) indexing: progress
  card with stage text, determinate bar, ETA, Cancel.
* **Import dialog**: file name, sheet combo (hidden when one sheet), header-row spin (auto-detected,
  1-based), preview table (first 30 rows under the header), then "How should each column be used?"
  with one row per column: name, muted sample values, segmented control **Deciding | Search |
  Display only** pre-set from `suggest_roles` with the suggestion reason as caption/tooltip.
  One-line explainer: "Deciding = identity that must match (brokerage, issuer, ISIN). Search = text
  that describes the item (fund name)." Footer: Cancel, primary "Import N rows". Validation: at least
  one Search or Deciding column. **Columns… dialog** = same role editor + learned-aliases info
  ("Learned automatically: iShares → BlackRock"), Apply rebuilds the index (cache makes it fast).
* **Drag & drop** a supported file anywhere → overlay "Drop to import" → import dialog.
* **Toasts** (bottom-right, auto-hide 4 s): "Exported 12 queries · Open folder", errors, "Index ready".
* **Status bar**: last message on the left; on the right a lock icon "Offline · data stays on this PC"
  and "Semantic search on" / "Lexical only (model missing)".
* Persisted UI settings: results_per_query, sort_mode, theme, window geometry, splitter sizes,
  details_visible, collapse state is not persisted (lists always start collapsed).

### 12.3 Threads
`QThreadPool` + `QRunnable` workers emitting through a `QObject` signals holder (`progress(str,
float)`, `done(object)`, `failed(str)`); cancellation through a `threading.Event` passed as the
`cancel` callback. Startup: show window immediately with history from the store, then load the
embedder and the stored dataset and rebuild the index in the background (embedding cache → seconds).
Only one index build at a time; a newer build supersedes an older one (cancel + restart). Never
block the UI thread for more than ~50 ms.

## 13. Security & privacy (R1)

* No sockets: no web server, no `QtNetwork`, no HTTP clients in the bundle (PyInstaller excludes
  `huggingface_hub`, `httpx*`, `requests`, `urllib3`, Qt network/QML/WebEngine modules and the `tls`
  and `networkinformation` Qt plugins). `HF_HUB_OFFLINE=1` set defensively at startup.
* Files are parsed by memory-safe calamine; no macros or formulas are executed.
* Exports never turn cell text into formulas (formula injection).
* All state is per-user in `%LOCALAPPDATA%\SemanticDBSearch`; the installer offers to delete it on
  uninstall. SQL is always parameterised. No pickle anywhere.

## 14. Packaging

`packaging/build.ps1` → fetch/verify model → icon + sample data → tests → PyInstaller one-folder,
windowed build (`packaging/semdb.spec`, name `SemanticDBSearch`, version resource, icon) → selftest of
the frozen exe (`--selftest`, exit code 0) → Inno Setup (`packaging/installer.iss`): per-user install
without admin rights (`PrivilegesRequired=lowest`, `{autopf}`), Start-menu shortcut, optional desktop
shortcut, launch after install, uninstaller that optionally removes user data, LZMA2 solid
compression. Output: `dist/SemanticDBSearch-Setup-<version>.exe`.

## 15. Quality bar

* `pytest` green; core modules have focused unit tests; one offscreen UI smoke test.
* `tools/evaluate.py` on the synthetic 45k-row fund universe reports Acc@1 / MRR@10 / brand
  precision for the full hybrid vs ablations; the hybrid must win, and brand precision@5 for queries
  that name a brand must be ≥ 0.95.
* Performance on a mid-range laptop: import 45k rows < 10 s; first index ≈ 1 min (cached afterwards:
  < 10 s); search < 50 ms per query.

"""Prepare the bundled sentence-embedding model: download, verify, quantize, check, install.

Run from the repository root with the development environment (quantization needs ``onnx``)::

    python tools/fetch_model.py                      # build models/paraphrase-multilingual-MiniLM-L12-v2
    python tools/fetch_model.py --check              # verify that folder offline, no network
    python tools/fetch_model.py --out build/model-test --force

Steps: download the pinned fp32 ONNX export, tokenizer and configuration files of
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (SHA-256 verified, cached under
build/model-download), quantize with onnxruntime's dynamic int8 recipe, compare quantized and
fp32 embeddings of fixed probe sentences, write manifest.json into a staging folder, and swap
it into place only when every check passed.  Only this development tool uses the network;
the app never does.  Exit status: 0 success, 1 failure.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import http.client
import json
import os
import platform
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = REPO.split("/")[1]
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
LICENSE = "Apache-2.0"
URL_TEMPLATE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
SOURCE_FILES = {  # path in the model repository -> pinned SHA-256
    "onnx/model.onnx": "10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a",
    "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
    "tokenizer_config.json": "5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b",
    "config.json": "6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e",
    "sentence_bert_config.json": "70f4448f31320443fe3557cacea5abf2dcc4915dda8c80646bec9f3bb0aa5a1f",
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "modules.json": "8f4b264b80206c830bebbdcae377e137925650a433b689343a63bdc9b3145460",
    "special_tokens_map.json": "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
}
QUANTIZATION = {
    "tag": "q8rr",  # part of the app's model_id, hence of every embedding-cache key
    "method": "onnxruntime.quantization.quantize_dynamic",
    "weight_type": "QInt8",
    "per_channel": True,
    "reduce_range": True,
}
OUTPUT_FILES = ("model.onnx", "tokenizer.json")
MANIFEST_FORMAT = 1
FIDELITY_THRESHOLD = 0.98  # minimum mean cosine between quantized and fp32 probe embeddings
USER_AGENT = "Fundscope-fetch-model/1.0 (+https://github.com/MDz1C/Semantic-DB-search)"
TIMEOUT_S = 60
ATTEMPTS = 3
CHUNK_BYTES = 1 << 20

# Fixed probe sentences for the fidelity check: fund names and financial phrases, English and Spanish.
PROBES = [
    "iShares Core S&P 500 UCITS ETF USD (Acc)",
    "Vanguard FTSE All-World UCITS ETF Distributing",
    "Amundi MSCI Emerging Markets UCITS ETF",
    "Xtrackers MSCI World UCITS ETF 1C",
    "SPDR Bloomberg 1-3 Year Euro Government Bond UCITS ETF",
    "Global aggregate bond index fund hedged to euro",
    "Emerging markets equity fund with low fees",
    "Short-term US Treasury bills money market fund",
    "Physical gold exchange traded commodity",
    "European dividend aristocrats equity income fund",
    "Technology sector ETF tracking the Nasdaq 100",
    "Investment grade corporate bonds denominated in euros",
    "High yield bond fund with monthly distributions",
    "Inflation-linked government bonds",
    "Small cap value stocks in the United States",
    "Sustainable ESG screened world equity fund",
    "Real estate investment trusts across developed markets",
    "Accumulating share class with a total expense ratio of 0.07%",
    "Japanese equities, currency hedged",
    "Balanced fund with 60% equities and 40% bonds",
    "Emerging Markets",
    "Government Bond",
    "Fondo de renta variable de mercados emergentes",
    "Fondo indexado al S&P 500, clase de acumulación",
    "Bonos del gobierno de la zona euro a corto plazo",
    "Fondo monetario en euros de bajo riesgo",
    "Renta fija corporativa con grado de inversión",
    "Fondo de inversión sostenible con criterios ESG",
    "Acciones de pequeñas empresas europeas",
    "Fondo mixto conservador con un 30 % en renta variable",
    "Gestora de fondos con sede en Madrid",
    "Comisión de gestión anual del 0,15 %",
    "Fondo cotizado que replica el índice MSCI World",
    "Bonos ligados a la inflación de Estados Unidos",
    "Fondo de dividendos de compañías españolas",
    "Sector tecnológico de Estados Unidos",
    "Oro físico como activo refugio",
    "Deuda pública de países emergentes en moneda local",
    "Fondo inmobiliario cotizado europeo",
    "mercados emergentes",
]


class FetchError(Exception):
    """A preparation step failed; the message is shown to the user."""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Piped output (build.ps1) is encoded in the ANSI code page: escape what it cannot show, such
    # as a non-Latin user name in a path, instead of crashing.  Line buffering keeps stdout and
    # stderr in order in build logs.
    sys.stdout.reconfigure(line_buffering=True, errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
    out = args.out.resolve()
    try:
        if args.check:
            return _report_check(out)
        if not args.force and not check_folder(out):
            print(f"Model already prepared and verified: {out} (use --force to rebuild)")
            return 0
        prepare(out, args.cache_dir.resolve())
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 1
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, verify and quantize the sentence-embedding model bundled with the app."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / MODEL_NAME,
        help=f"model folder to create or check (default: models/{MODEL_NAME})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "build" / "model-download",
        help="where downloads are kept and reused when their hashes match (default: build/model-download)",
    )
    parser.add_argument(
        "--check", action="store_true", help="only verify --out (manifest present, hashes match); no network"
    )
    parser.add_argument("--force", action="store_true", help="rebuild even when --out is already valid")
    return parser.parse_args(argv)


# Preparation ------------------------------------------------------------------------------


def prepare(out: Path, cache_dir: Path) -> None:
    """Build a verified model folder at ``out`` (replacing it only after every check passed)."""
    started = time.monotonic()
    source = cache_dir / REVISION
    print(f"Fetching {REPO} @ {REVISION[:7]} into {source}")
    for path, sha256 in SOURCE_FILES.items():
        fetch(path, sha256, source / path)
    settings = _model_settings(source)

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f".{out.name}.staging-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        print("Quantizing (dynamic int8, per-channel, 7-bit weights)...")
        step = time.monotonic()
        _quantize(source / "onnx" / "model.onnx", staging / "model.onnx")
        shutil.copyfile(source / "tokenizer.json", staging / "tokenizer.json")
        size_mb = (staging / "model.onnx").stat().st_size / 1e6
        print(f"  model.onnx: {size_mb:.1f} MB in {time.monotonic() - step:.1f} s")

        print(f"Checking fidelity on {len(PROBES)} probe sentences...")
        fidelity = _fidelity(source / "onnx" / "model.onnx", staging, settings)
        print(
            f"  cosine to fp32: mean {fidelity['mean_cosine']:.4f}, min {fidelity['min_cosine']:.4f}"
            f" (required mean >= {FIDELITY_THRESHOLD})"
        )
        if fidelity["mean_cosine"] < FIDELITY_THRESHOLD:
            raise FetchError(
                f"the quantized model is not faithful enough (mean cosine {fidelity['mean_cosine']:.4f}"
                f" < {FIDELITY_THRESHOLD}); {out} was left unchanged"
            )

        _write_manifest(staging, settings, fidelity)
        problems = check_folder(staging)
        if problems:
            raise FetchError("the new model folder failed verification: " + "; ".join(problems))
        _swap_into_place(staging, out)
        _remove_leftovers(out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    print(f"Done in {time.monotonic() - started:.0f} s: {out}")
    for name, info in manifest["files"].items():
        print(f"  {name}  sha256 {info['sha256']}  ({info['bytes']:,} bytes)")


def fetch(path: str, sha256: str, dest: Path) -> None:
    """Make ``dest`` hold the pinned file ``path``, reusing a cached copy whose hash matches."""
    if dest.is_file():
        if _sha256(dest) == sha256:
            print(f"  cached      {path}")
            return
        print(f"  stale       {path} (hash mismatch), downloading again")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = URL_TEMPLATE.format(repo=REPO, revision=REVISION, path=path)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _download(url, part, path)
            digest = _sha256(part)
            if digest != sha256:
                part.unlink()
                raise FetchError(f"SHA-256 {digest} does not match the pinned {sha256}")
            os.replace(part, dest)
            return
        except (FetchError, OSError, http.client.HTTPException) as exc:
            if attempt == ATTEMPTS:
                raise FetchError(f"could not download {path} from {url}: {exc}") from exc
            delay = 3 * attempt
            print(f"  attempt {attempt}/{ATTEMPTS} for {path} failed ({exc}); retry in {delay} s", file=sys.stderr)
            time.sleep(delay)


def _download(url: str, part: Path, label: str) -> None:
    """Stream ``url`` into ``part``, resuming a partial file when the server supports ranges."""
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:  # nothing left to fetch: the hash check decides
            exc.close()
            return
        raise
    with response:
        if offset and response.status != 206:
            offset = 0  # the server ignored the range and sends the whole file
        length = response.headers.get("Content-Length", "")
        total = offset + int(length) if length.isdigit() else None
        meter = _Meter(label, total, offset)
        with open(part, "ab" if offset else "wb") as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                meter.advance(len(chunk))
        meter.finish()
    if total is not None and part.stat().st_size != total:
        raise FetchError(f"connection closed after {part.stat().st_size:,} of {total:,} bytes")


class _Meter:
    """Download progress: a self-updating line on a terminal, one line per 10 % otherwise."""

    def __init__(self, label: str, total: int | None, done: int) -> None:
        self.label, self.total, self.done = label, total, done
        self.resumed = done
        self.started = time.monotonic()
        self.shown_at = 0.0
        self.width = 0
        self.tty = sys.stdout.isatty()
        self.decile = self._decile()
        if done:
            print(f"  resuming    {label} at {_size(done)}")

    def _decile(self) -> int:
        return self.done * 10 // self.total if self.total else 0

    def _line(self, final: bool) -> str:
        line = f"  download    {self.label}  {_size(self.done)}"
        if self.total and not final:
            line += f" of {_size(self.total)} ({100 * self.done // self.total} %)"
        fetched = self.done - self.resumed
        if fetched >= 1e6:
            line += f"  {fetched / 1e6 / max(time.monotonic() - self.started, 1e-6):.1f} MB/s"
        return line

    def _show(self, line: str, end: str) -> None:
        if self.tty:
            line = "\r" + line.ljust(self.width)
            self.width = max(self.width, len(line) - 1)
        print(line, end=end, flush=True)

    def advance(self, size: int) -> None:
        self.done += size
        if self.tty:
            now = time.monotonic()
            if now - self.shown_at >= 0.25:
                self.shown_at = now
                self._show(self._line(final=False), end="")
        elif self.total and self.done < self.total and self._decile() > self.decile:
            self.decile = self._decile()
            self._show(self._line(final=False), end="\n")

    def finish(self) -> None:
        self._show(self._line(final=True), end="\n")


def _model_settings(source: Path) -> dict:
    """dim / max_seq_length / pooling from the upstream sentence-transformers configuration,
    refusing configurations the app's runtime (mean pooling, no lower-casing) does not implement."""
    modules = _load_json(source / "modules.json")
    kinds = [module.get("type", "") for module in modules]
    supported = (".Transformer", ".Pooling", ".Normalize")
    if not kinds or any(not kind.endswith(supported) for kind in kinds):
        raise FetchError(f"unsupported sentence-transformers modules {kinds}")
    pooling = _load_json(source / "1_Pooling" / "config.json")
    modes = sorted(key for key, value in pooling.items() if key.startswith("pooling_mode_") and value is True)
    if modes != ["pooling_mode_mean_tokens"]:
        raise FetchError(f"the app implements mean pooling only, the model uses {modes}")
    sbert = _load_json(source / "sentence_bert_config.json")
    if sbert.get("do_lower_case"):
        raise FetchError("the model expects lower-cased input, which the app does not produce")
    dim = pooling.get("word_embedding_dimension")
    hidden = _load_json(source / "config.json").get("hidden_size")
    max_seq_length = sbert.get("max_seq_length")
    if not isinstance(dim, int) or dim != hidden or not isinstance(max_seq_length, int) or max_seq_length < 8:
        raise FetchError(
            f"inconsistent model configuration (dim {dim}, hidden {hidden}, max_seq_length {max_seq_length})"
        )
    return {"dim": dim, "max_seq_length": max_seq_length, "pooling": "mean"}


def _quantize(source: Path, target: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        source,
        target,
        weight_type=QuantType[QUANTIZATION["weight_type"]],
        per_channel=QUANTIZATION["per_channel"],
        reduce_range=QUANTIZATION["reduce_range"],
    )


def _fidelity(reference_model: Path, folder: Path, settings: dict) -> dict:
    """Cosine similarity between quantized and fp32 embeddings of every probe sentence."""
    tokenizer = folder / "tokenizer.json"
    reference = _embed(reference_model, tokenizer, PROBES, settings["max_seq_length"])
    quantized = _embed(folder / "model.onnx", tokenizer, PROBES, settings["max_seq_length"])
    if quantized.shape != (len(PROBES), settings["dim"]):
        raise FetchError(f"the quantized model produced embeddings of shape {quantized.shape}")
    cosines = (reference * quantized).sum(axis=1)
    return {
        "probes": len(PROBES),
        "mean_cosine": round(float(cosines.mean()), 5),
        "min_cosine": round(float(cosines.min()), 5),
        "threshold": FIDELITY_THRESHOLD,
    }


def _embed(model: Path, tokenizer_file: Path, texts: list[str], max_seq_length: int):
    """Mean-pooled, L2-normalized embeddings, one text per run exactly like the app."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.enable_truncation(max_length=max_seq_length)
    tokenizer.no_padding()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    rows = []
    for encoding in tokenizer.encode_batch(texts):
        ids = np.asarray([encoding.ids], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
        vector = session.run(None, feeds)[0][0].mean(axis=0)
        rows.append(vector / np.linalg.norm(vector))
    return np.stack(rows)


def _write_manifest(folder: Path, settings: dict, fidelity: dict) -> None:
    import onnx
    import onnxruntime

    manifest = {
        "format": MANIFEST_FORMAT,
        "model": MODEL_NAME,
        "repo": REPO,
        "revision": REVISION,
        "license": LICENSE,
        "source": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "source_sha256": SOURCE_FILES,
        "quantization": QUANTIZATION,
        "tools": {
            "onnxruntime": onnxruntime.__version__,
            "onnx": onnx.__version__,
            "python": platform.python_version(),
        },
        **settings,
        "normalized": True,
        "fidelity": fidelity,
        "files": {
            name: {"sha256": _sha256(folder / name), "bytes": (folder / name).stat().st_size}
            for name in OUTPUT_FILES
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _swap_into_place(staging: Path, out: Path) -> None:
    """Replace ``out`` by ``staging``; the previous folder is restored if the move fails."""
    previous = out.with_name(f".{out.name}.previous-{os.getpid()}")
    if out.exists():
        shutil.rmtree(previous, ignore_errors=True)  # left by an interrupted run: a folder cannot be replaced
        try:
            os.replace(out, previous)
        except OSError as exc:
            raise FetchError(f"cannot replace {out} ({exc}); close programs using the model and retry") from exc
    try:
        os.replace(staging, out)
    except OSError as exc:
        if previous.exists():
            os.replace(previous, out)
        raise FetchError(f"cannot move the new model into {out}: {exc}") from exc
    shutil.rmtree(previous, ignore_errors=True)


def _remove_leftovers(out: Path) -> None:
    """Delete the staging and previous-version folders that killed runs left next to ``out``
    (only after a successful swap: until then they may hold the last good model)."""
    for kind in ("staging", "previous"):
        for leftover in out.parent.glob(f"{glob.escape('.' + out.name)}.{kind}-*"):
            shutil.rmtree(leftover, ignore_errors=True)


# Verification -----------------------------------------------------------------------------


def check_folder(folder: Path) -> list[str]:
    """Problems with a prepared model folder; an empty list means it is complete and intact."""
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path} not found"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{manifest_path} is unreadable: {exc}"]
    if not isinstance(manifest, dict):
        return [f"{manifest_path} is not a JSON object"]

    problems = []
    for key, expected in (("repo", REPO), ("revision", REVISION), ("quantization", QUANTIZATION)):
        if manifest.get(key) != expected:
            problems.append(f"manifest {key} is {manifest.get(key)!r}, expected {expected!r}")
    fidelity = manifest.get("fidelity")
    mean = fidelity.get("mean_cosine") if isinstance(fidelity, dict) else None
    if not isinstance(mean, (int, float)) or mean < FIDELITY_THRESHOLD:
        problems.append("manifest records no passing fidelity check")
    files = manifest.get("files")
    if not isinstance(files, dict):
        return problems + ["manifest lists no files"]
    problems += [f"manifest does not list {name}" for name in OUTPUT_FILES if name not in files]
    for name, info in files.items():
        path = folder / name
        if Path(name).name != name or not isinstance(info, dict):
            problems.append(f"manifest has an invalid file entry {name!r}")
        elif not path.is_file():
            problems.append(f"{name} is missing")
        elif path.stat().st_size != info.get("bytes"):
            problems.append(f"{name} is {path.stat().st_size} bytes, the manifest says {info.get('bytes')!r}")
        elif _sha256(path) != info.get("sha256"):
            problems.append(f"{name} does not match the SHA-256 recorded in the manifest")
    return problems


def _report_check(folder: Path) -> int:
    problems = check_folder(folder)
    if problems:
        print(f"FAILED: {folder}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    fidelity = manifest["fidelity"]
    print(f"OK: {folder}")
    print(f"  {manifest['repo']} @ {manifest['revision'][:7]}, quantization {manifest['quantization']['tag']}")
    print(f"  fidelity vs fp32: mean cosine {fidelity['mean_cosine']}, min {fidelity.get('min_cosine')}")
    for name, info in manifest["files"].items():
        print(f"  {name}  sha256 {info['sha256']}")
    return 0


def _size(count: int) -> str:
    return f"{count / 1e6:.1f} MB" if count >= 1e5 else f"{count / 1e3:.1f} KB"


def _sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"cannot read {path}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())

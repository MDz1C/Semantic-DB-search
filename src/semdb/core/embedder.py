"""Offline sentence embeddings with ONNX Runtime + HF tokenizers (no PyTorch, no network).

Every text is embedded in its own single-threaded model run, and the runs of different texts
proceed in parallel on a worker pool.  The bundled model quantizes activations dynamically,
with one scale per tensor, so inside a padded batch a text's vector would depend on its batch
neighbours.  One text per run makes each vector a pure function of its text (a row and an
identical query, alone or in a list, always get the same vector) and, spread over the CPU
cores, it is also faster than padded batches from four logical CPUs up.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import paths
from .types import CancelFn, Cancelled

log = logging.getLogger(__name__)

MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
QUANTIZATION_TAG = "q8rr"
DEFAULT_MAX_SEQ_LENGTH = 128
_MAX_POSITIONS = 512  # position embeddings of the XLM-R based architecture
_CHARS_PER_TOKEN = 32  # texts are cut to max_seq_length * this many characters before tokenizing
_TOKENIZE_CHUNK = 4096
_MAX_WORKERS = 32
_INPUTS = ("input_ids", "attention_mask", "token_type_ids")
_WORD = re.compile(r"\S+")


class ModelMissingError(Exception):
    """The bundled model folder is missing or incomplete."""


class OnnxEmbedder:
    """Mean-pooled, L2-normalized sentence embeddings. Thread-safe ``encode``.

    ``threads`` texts are embedded concurrently (default: one per logical CPU, at most 32);
    ``batch_size`` is the number of texts between two progress reports.
    """

    model_id: str
    dim: int

    def __init__(self, model_dir: str | Path, batch_size: int = 64, threads: int | None = None) -> None:
        folder = Path(model_dir)
        missing = [name for name in ("model.onnx", "tokenizer.json") if not (folder / name).is_file()]
        if missing:
            raise ModelMissingError(f"{folder} does not contain {' and '.join(missing)}")
        manifest = _read_manifest(folder / "manifest.json")
        name = _setting(manifest, "model", paths.MODEL_NAME, lambda v: isinstance(v, str) and bool(v.strip()))
        revision = _setting(manifest, "revision", MODEL_REVISION, lambda v: isinstance(v, str) and len(v) >= 7)
        quantization = _setting(manifest, "quantization", {}, lambda v: isinstance(v, dict))
        tag = quantization.get("tag", QUANTIZATION_TAG)
        _setting(manifest, "pooling", "mean", lambda v: v == "mean")
        self.max_seq_length: int = _setting(
            manifest,
            "max_seq_length",
            DEFAULT_MAX_SEQ_LENGTH,
            lambda v: type(v) is int and 2 < v <= _MAX_POSITIONS,
        )
        self.model_id = f"{name}@{revision[:7]}/{tag}"
        self.batch_size = max(1, batch_size)
        self.threads = max(1, threads) if threads else _default_threads()

        self._tokenizer = _load_tokenizer(folder / "tokenizer.json", self.max_seq_length)
        self._tokenizer_lock = threading.Lock()  # tokenizers.Tokenizer is not documented as thread-safe
        self._session, inputs, self._output = _load_session(folder / "model.onnx")
        self._token_types = "token_type_ids" in inputs
        try:
            probe = self._embed_ids(self._token_ids(["Emerging markets equity fund"])[0])
        except Exception as exc:  # onnxruntime raises its own exception types
            raise ModelMissingError(f"model.onnx failed a test run: {exc}") from exc
        self.dim = int(probe.shape[0])
        if self.dim != manifest.get("dim", self.dim):
            raise ModelMissingError(
                f"model.onnx gives {self.dim}-d embeddings, manifest.json says {manifest['dim']!r}"
            )
        if not np.isfinite(probe).all():
            raise ModelMissingError("model.onnx produced a non-finite test embedding")
        self._pool: ThreadPoolExecutor | None = None
        if self.threads > 1:
            self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="semdb-embed")
        self._window = 4 * self.threads  # texts queued ahead so no worker idles while results drain in order

    def encode(
        self,
        texts: list[str],
        progress: Callable[[int, int], None] | None = None,
        cancel: CancelFn | None = None,
    ) -> np.ndarray:
        """Return float32 array (len(texts), dim), rows L2-normalized, in input order.

        ``progress(done, total)`` is called with ``(0, total)`` first, then every ``batch_size``
        texts and once at the end (never for an empty list).  ``cancel`` is polled before every
        text; when it returns True, ``Cancelled`` is raised.
        """
        texts = list(texts)
        total = len(texts)
        vectors = np.empty((total, self.dim), dtype=np.float32)
        if not total:
            return vectors
        if progress is not None:
            progress(0, total)
        queue = iter(self._token_ids(texts, cancel))
        pool = self._pool if total > 1 else None
        pending: deque[Future[np.ndarray]] = deque()
        try:
            for done in range(1, total + 1):
                _check(cancel)
                if pool is None:
                    vectors[done - 1] = self._embed_ids(next(queue))
                else:
                    refill = islice(queue, self._window - len(pending))
                    pending.extend(pool.submit(self._embed_ids, ids) for ids in refill)
                    vectors[done - 1] = pending.popleft().result()
                if progress is not None and (done % self.batch_size == 0 or done == total):
                    progress(done, total)
        finally:
            for future in pending:
                future.cancel()
        return vectors

    def _token_ids(self, texts: list[str], cancel: CancelFn | None = None) -> list[np.ndarray]:
        """Token ids per text, truncated to ``max_seq_length`` (special tokens included)."""
        limit = self.max_seq_length * _CHARS_PER_TOKEN  # bounds the tokenizing cost of huge cells
        ids: list[np.ndarray] = []
        for start in range(0, len(texts), _TOKENIZE_CHUNK):
            _check(cancel)
            chunk = [_clip(text, limit) for text in texts[start : start + _TOKENIZE_CHUNK]]
            with self._tokenizer_lock:
                try:
                    encodings = self._tokenizer.encode_batch(chunk)
                except TypeError:  # lone surrogates cannot be passed to the tokenizer: replace them
                    encodings = self._tokenizer.encode_batch([_valid_unicode(text) for text in chunk])
            ids.extend(np.array(encoding.ids, dtype=np.int64) for encoding in encodings)
        return ids

    def _embed_ids(self, ids: np.ndarray) -> np.ndarray:
        """Embedding of one tokenized text: an unpadded run, so mean pooling is a plain mean."""
        batch = ids[np.newaxis]
        feeds = {"input_ids": batch, "attention_mask": np.ones_like(batch)}
        if self._token_types:
            feeds["token_type_ids"] = np.zeros_like(batch)
        vector = self._session.run([self._output], feeds)[0][0].mean(axis=0)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


def load_default_embedder() -> OnnxEmbedder | None:
    """The bundled model, or None (logged) when it is unavailable -> lexical-only search."""
    folder = paths.model_dir()
    started = time.perf_counter()
    try:
        embedder = OnnxEmbedder(folder)
    except ModelMissingError as exc:
        log.warning("Semantic search disabled: %s", exc)
        return None
    except Exception:
        log.exception("Semantic search disabled: the embedding model in %s could not be loaded", folder)
        return None
    log.info(
        "Embedding model %s loaded from %s in %.1f s (%d workers)",
        embedder.model_id,
        folder,
        time.perf_counter() - started,
        embedder.threads,
    )
    return embedder


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))  # tolerate a BOM from Windows editors
    except (OSError, ValueError) as exc:
        raise ModelMissingError(f"{path} is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ModelMissingError(f"{path} is not a JSON object")
    return manifest


def _setting(manifest: dict[str, Any], key: str, default: Any, valid: Callable[[Any], bool]) -> Any:
    value = manifest.get(key, default)
    if not valid(value):
        raise ModelMissingError(f"manifest.json has an unsupported {key!r}: {value!r}")
    return value


def _load_tokenizer(path: Path, max_seq_length: int) -> Any:
    from tokenizers import Tokenizer

    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as exc:  # the Rust side raises plain Exception
        raise ModelMissingError(f"{path.name} could not be loaded: {exc}") from exc
    tokenizer.enable_truncation(max_length=max_seq_length)
    tokenizer.no_padding()
    return tokenizer


def _load_session(path: Path) -> tuple[Any, tuple[str, ...], str]:
    import onnxruntime as ort  # imported here: it takes ~0.5 s and callers load the model off the UI thread

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1  # parallelism comes from concurrent single-text runs
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime raises its own exception types for unreadable models
        raise ModelMissingError(f"{path.name} could not be loaded: {exc}") from exc
    inputs = {node.name: node.type for node in session.get_inputs()}
    if (
        not {"input_ids", "attention_mask"} <= inputs.keys() <= set(_INPUTS)
        or set(inputs.values()) != {"tensor(int64)"}
    ):
        raise ModelMissingError(
            f"{path.name} has inputs {inputs}; expected int64 input_ids, attention_mask (and token_type_ids)"
        )
    outputs = session.get_outputs()
    if not outputs or len(outputs[0].shape) != 3:
        raise ModelMissingError(f"{path.name} does not output one embedding per token")
    return session, tuple(inputs), outputs[0].name


def _default_threads() -> int:
    count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return max(1, min(count, _MAX_WORKERS))


def _check(cancel: CancelFn | None) -> None:
    if cancel is not None and cancel():
        raise Cancelled()


def _clip(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters.  Longer texts get their whitespace runs collapsed
    first (the tokenizer drops whitespace anyway), so padding cannot crowd out the words."""
    if len(text) <= limit:
        return text
    words: list[str] = []
    size = 0
    for match in _WORD.finditer(text):  # stops once enough words are collected
        words.append(match.group())
        size += len(words[-1]) + 1
        if size > limit:
            break
    return " ".join(words)[:limit]


def _valid_unicode(text: str) -> str:
    return text.encode("utf-8", "replace").decode("utf-8") if isinstance(text, str) else text

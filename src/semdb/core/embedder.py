"""Offline sentence embeddings with ONNX Runtime + HF tokenizers (no PyTorch, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .types import CancelFn


class ModelMissingError(Exception):
    """The bundled model folder is missing or incomplete."""


class OnnxEmbedder:
    """Mean-pooled, L2-normalized sentence embeddings. Thread-safe ``encode``."""

    model_id: str
    dim: int

    def __init__(self, model_dir: str | Path, batch_size: int = 64, threads: int | None = None) -> None:
        raise NotImplementedError

    def encode(
        self,
        texts: list[str],
        progress: Callable[[int, int], None] | None = None,
        cancel: CancelFn | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


def load_default_embedder() -> OnnxEmbedder | None:
    """The bundled model, or None (logged) when it is unavailable -> lexical-only search."""
    raise NotImplementedError

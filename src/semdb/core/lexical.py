"""Character n-gram + word TF-IDF index (numpy only), for exact-token and typo-tolerant matching."""

from __future__ import annotations

import numpy as np

from .types import CancelFn


class LexicalIndex:
    """Cosine similarity between a query and every document, via an inverted index."""

    def __init__(self, documents: list[str], cancel: CancelFn | None = None) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def scores(self, query: str) -> np.ndarray:
        """float32 array (n_documents,) of cosine similarities in [0, 1]."""
        raise NotImplementedError

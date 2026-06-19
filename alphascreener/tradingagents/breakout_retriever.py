"""Breakout case retrieval via faiss + cases.parquet.

Issue #97: Analyst prompts + invocation.
Reference: PRD 4.2.1 — Breakout Analyst 历史相似案例检索.

Maintains a faiss index over historical positive-sample factor vectors
stored in ``~/.alphascreener/data/case_library/cases.parquet``.
Supports cosine-similarity nearest-neighbour search (faiss.IndexFlatIP
when < 50K vectors; faiss.IndexHNSWFlat when >= 50K).

MVP behaviour: when the case library is empty, ``search()`` returns ``[]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import polars as pl

from alphascreener.logging import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CASE_LIBRARY_DIR: Path = Path.home() / ".alphascreener" / "data" / "case_library"
_CASES_PARQUET: Path = _CASE_LIBRARY_DIR / "cases.parquet"
_INDEX_FILE: Path = _CASE_LIBRARY_DIR / "faiss.index"
_DEFAULT_TOPK: int = 5
_SIMILARITY_CUTOFF: float = 0.85
_TRANSITION_THRESHOLD: int = 50_000


_logger = get_logger("screening")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class BreakoutCaseRetriever:
    """Faiss-powered cosine-similarity search over historical breakout cases.

    Usage::

        retriever = BreakoutCaseRetriever()
        results = retriever.search(factor_vector=[0.12, -0.03, ...], top_k=5)
        # results == [{"ticker": "AAPL", "date": "2023-05-15",
        #               "similarity": 0.91, "actual_pnl": 0.135}, ...]
    """

    def __init__(
        self,
        index_path: Path | None = None,
        parquet_path: Path | None = None,
    ) -> None:
        self._index_path = index_path or _INDEX_FILE
        self._parquet_path = parquet_path or _CASES_PARQUET
        self._index: faiss.Index | None = None
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] | None = None
        self._initialized: bool = False

    # -- public API ------------------------------------------------------------

    def search(
        self,
        factor_vector: list[float],
        top_k: int = _DEFAULT_TOPK,
        similarity_cutoff: float = _SIMILARITY_CUTOFF,
    ) -> list[dict[str, Any]]:
        """Find top-k similar historical breakout cases.

        Args:
            factor_vector: The query factor vector.
            top_k: Number of neighbours to retrieve.
            similarity_cutoff: Minimum cosine similarity (inner-product
                on L2-normalised vectors) to include in results.

        Returns:
            List of dicts ``{"ticker", "date", "similarity", "actual_pnl"}``.
            Empty list when the case library has no vectors or no match
            meets the cutoff.
        """
        self._ensure_initialized()

        if self._index is None or self._index.ntotal == 0:
            _logger.debug("Breakout case library is empty — returning []")
            return []

        query = np.array(factor_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._index.d:
            _logger.debug(
                "Query dim %d != index dim %d — returning []", query.shape[1], self._index.d
            )
            return []
        faiss.normalize_L2(query)
        distances, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._metadata or []):
                continue
            if float(dist) < similarity_cutoff:
                continue
            meta = (self._metadata or [])[idx]
            results.append(
                {
                    "ticker": meta.get("ticker", ""),
                    "date": meta.get("date", ""),
                    "similarity": round(float(dist), 4),
                    "actual_pnl": meta.get("actual_pnl", 0.0),
                }
            )
        return results

    def has_cases(self) -> bool:
        """Return ``True`` if the case library contains at least one entry."""
        self._ensure_initialized()
        return self._index is not None and self._index.ntotal > 0

    # -- internal --------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Lazy-initialise the faiss index from cases.parquet."""
        if self._initialized:
            return
        self._initialized = True

        self._parquet_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._parquet_path.exists():
            _logger.debug("cases.parquet not found at %s — empty library", self._parquet_path)
            return

        try:
            df = pl.read_parquet(str(self._parquet_path))
        except Exception:
            _logger.warning("Failed to read cases.parquet — treating as empty", exc_info=True)
            return

        if df.is_empty():
            return

        # Extract factor vector columns — columns prefixed "f_"
        factor_cols = sorted(c for c in df.columns if c.startswith("f_"))
        if not factor_cols:
            _logger.warning("cases.parquet has no factor columns — skipping index build")
            return

        vectors = np.ascontiguousarray(df.select(factor_cols).to_numpy().astype(np.float32))
        n_vectors, dim = vectors.shape
        _logger.debug("Building faiss index: %d vectors, %d dims", n_vectors, dim)

        # Build metadata list
        meta_keys = ["ticker", "date", "actual_pnl"]
        metadata: list[dict[str, Any]] = []
        for i in range(n_vectors):
            row: dict[str, Any] = {}
            for key in meta_keys:
                if key in df.columns:
                    row[key] = df[key][i]
                else:
                    row[key] = "" if key != "actual_pnl" else 0.0
            metadata.append(row)

        # L2-normalise for cosine similarity (faiss inner product = cosine on normed vectors)
        faiss.normalize_L2(vectors)

        # Choose index type based on size
        if n_vectors < _TRANSITION_THRESHOLD:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexHNSWFlat(dim, 32)
        index.add(vectors)

        self._index = index
        self._vectors = vectors
        self._metadata = metadata

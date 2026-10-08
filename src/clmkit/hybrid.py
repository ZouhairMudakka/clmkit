"""Small in-memory BM25 and exact dense/BM25 reciprocal rank fusion.

Only NumPy is required. These retrievers own their corpus; all mutations go
through add/update/delete. Writes are staged before activation but are neither
durable nor safe for concurrent readers/writers. Rebuild after changing encoder
weights or formatting. Persistence and approximate indexes are not supported.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, overload

import numpy as np

from clmkit.encoders.base import Encoder
from clmkit.retrieval import Document, MetadataFilter
from clmkit.types import SearchHit


def _tokens(text: str) -> list[str]:
    # Fixed Unicode word tokenizer: lowercase, no stemming or stopword removal.
    return re.findall(r"\w+", text.lower())


@dataclass
class _Corpus:
    documents: dict[str, Document]
    ids: list[str]
    lengths: np.ndarray
    postings: dict[str, tuple[np.ndarray, np.ndarray]]


class BM25Retriever:
    """BM25 with positive Robertson IDF and a fixed Unicode word tokenizer.

    Query terms are counted once. Corpus statistics always use the complete
    corpus, even for filtered searches. Only positive lexical matches are
    returned; empty/unknown queries return no matches. Ties use ascending ID.
    Metadata is copied on input and output. Scores are not probabilities.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if not np.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and > 0")
        if not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between 0 and 1")
        self.k1 = float(k1)
        self.b = float(b)
        self._corpus = self._build({})

    @staticmethod
    def _build(documents: dict[str, Document]) -> _Corpus:
        ids = sorted(documents)
        lengths = np.zeros(len(ids), dtype=np.float64)
        postings: dict[str, list[tuple[int, int]]] = {}
        for position, id_ in enumerate(ids):
            terms = Counter(_tokens(documents[id_].text))
            lengths[position] = sum(terms.values())
            for term, count in terms.items():
                postings.setdefault(term, []).append((position, count))
        packed = {}
        for term, values in postings.items():
            rows, counts = zip(*values, strict=True)
            packed[term] = (np.asarray(rows, dtype=np.intp), np.asarray(counts, dtype=np.float64))
        return _Corpus(documents, ids, lengths, packed)

    def _replace(self, documents: dict[str, Document], changed: list[str], batch_size: int) -> None:
        self._corpus = self._build(documents)

    def __len__(self) -> int:
        return len(self._corpus.ids)

    def __contains__(self, id_: object) -> bool:
        return id_ in self._corpus.documents

    def get(self, id_: str) -> Document | None:
        """Return a detached document; edit through update(), not this copy."""
        return deepcopy(self._corpus.documents.get(id_))

    @staticmethod
    def _document(text: str, metadata: Mapping[str, Any] | None) -> Document:
        if not isinstance(text, str):
            raise TypeError("texts must be strings")
        meta = deepcopy(dict(metadata or {}))
        json.dumps(meta, allow_nan=False)
        return Document(text, meta)

    def add(
        self,
        texts: Sequence[str],
        *,
        ids: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        """Add documents, rejecting duplicate IDs before changing either index."""
        if isinstance(texts, str):
            raise TypeError("add() expects a sequence of strings")
        texts = list(texts)
        new_ids = [uuid.uuid4().hex for _ in texts] if ids is None else list(ids)
        metas = [None] * len(texts) if metadata is None else list(metadata)
        if not len(texts) == len(new_ids) == len(metas):
            raise ValueError("texts, ids and metadata must have the same length")
        if any(not isinstance(i, str) or not i for i in new_ids):
            raise TypeError("ids must be non-empty strings")
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("duplicate ids within one add() call")
        if any(i in self for i in new_ids):
            raise ValueError("ids already in index")
        documents = dict(self._corpus.documents)
        for id_, text, meta in zip(new_ids, texts, metas, strict=True):
            documents[id_] = self._document(text, meta)
        if new_ids:
            self._replace(documents, new_ids, batch_size)
        return new_ids

    def update(
        self,
        id_: str,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        batch_size: int = 32,
    ) -> None:
        """Replace an existing text; omitted metadata retains existing metadata.

        BM25 statistics are rebuilt; hybrid only re-encodes the changed document.
        A validation/encoding failure leaves the previous corpus active.
        """
        if id_ not in self:
            raise KeyError(id_)
        documents = dict(self._corpus.documents)
        meta = documents[id_].metadata if metadata is None else metadata
        documents[id_] = self._document(text, meta)
        self._replace(documents, [id_], batch_size)

    def delete(self, ids: Sequence[str]) -> int:
        """Remove IDs from both candidate sources and refresh BM25 statistics."""
        documents = dict(self._corpus.documents)
        for id_ in ids:
            documents.pop(id_, None)
        count = len(self) - len(documents)
        if count:
            self._replace(documents, [], 32)
        return count

    def _eligible(self, filter: MetadataFilter | None) -> np.ndarray:
        eligible = []
        for position, id_ in enumerate(self._corpus.ids):
            meta = self._corpus.documents[id_].metadata
            if filter is None:
                matches = True
            elif callable(filter):
                matches = bool(filter(deepcopy(meta)))
            else:
                matches = all(meta.get(key) == value for key, value in filter.items())
            if matches:
                eligible.append(position)
        return np.asarray(eligible, dtype=np.intp)

    def _scores(self, query: str) -> np.ndarray:
        corpus = self._corpus
        scores = np.zeros(len(self), dtype=np.float64)
        avg_length = corpus.lengths.mean() if len(self) else 0.0
        if avg_length == 0:
            return scores
        # Sorting terms also fixes floating point accumulation order across runs.
        for term in sorted(set(_tokens(query))):
            posting = corpus.postings.get(term)
            if posting is None:
                continue
            rows, frequency = posting
            idf = np.log1p((len(self) - len(rows) + 0.5) / (len(rows) + 0.5))
            norm = self.k1 * (1 - self.b + self.b * corpus.lengths[rows] / avg_length)
            scores[rows] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return scores

    @staticmethod
    def _rank(scores: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
        # Positions follow sorted IDs; lexical and dense ties have one policy.
        return eligible[np.lexsort((eligible, -scores[eligible]))[:k]]

    def _hit(self, position: int, score: float) -> SearchHit:
        id_ = self._corpus.ids[position]
        document = self._corpus.documents[id_]
        return SearchHit(id_, float(score), document.text, deepcopy(document.metadata))

    def _search_batch(self, queries: list[str], k: int, eligible: np.ndarray) -> list[list[SearchHit]]:
        result = []
        for query in queries:
            scores = self._scores(query)
            candidates = eligible[scores[eligible] > 0]
            result.append([self._hit(int(i), scores[i]) for i in self._rank(scores, candidates, k)])
        return result

    @overload
    def search(self, query: str, k: int = ..., *, filter: MetadataFilter | None = ...) -> list[SearchHit]: ...

    @overload
    def search(
        self, query: list[str], k: int = ..., *, filter: MetadataFilter | None = ...
    ) -> list[list[SearchHit]]: ...

    def search(
        self, query: str | Sequence[str], k: int = 5, *, filter: MetadataFilter | None = None
    ) -> list[SearchHit] | list[list[SearchHit]]:
        """Rank one query or a batch with exact metadata eligibility.

        A mapping requires equality for each field; use a predicate for ranges
        or compound constraints. Predicates run once per document per call.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be an integer >= 1")
        single = isinstance(query, str)
        queries = [query] if isinstance(query, str) else list(query)
        if any(not isinstance(q, str) for q in queries):
            raise TypeError("search() expects strings")
        eligible = self._eligible(filter)
        if not queries or not eligible.size:
            results: list[list[SearchHit]] = [[] for _ in queries]
        else:
            results = self._search_batch(queries, k, eligible)
        return results[0] if single else results


class HybridRetriever(BM25Retriever):
    """Exact cosine dense retrieval fused with BM25 by reciprocal rank.

    For each branch retain up to max(k, candidate_depth) eligible candidates.
    Fuse with score sum(1 / (rrf_k + one_based_rank)), then sort by score and
    ascending ID. Lexical zero matches contribute nothing; dense still ranks
    all eligible documents. Branch scores are not mixed or calibrated.

    Dense search computes one query-by-corpus vector at a time and filters before
    selecting candidates, so selective constraints do not underfill an otherwise
    populated dense branch. No approximate search backend is used. Mutation
    rebuilds lexical statistics but reuses unchanged document embeddings.
    """

    def __init__(
        self,
        encoder: Encoder,
        *,
        candidate_depth: int = 100,
        rrf_k: float = 60.0,
        k1: float = 1.5,
        b: float = 0.75,
        query_instruction: str | None = None,
    ) -> None:
        super().__init__(k1=k1, b=b)
        if isinstance(candidate_depth, bool) or not isinstance(candidate_depth, int) or candidate_depth < 1:
            raise ValueError("candidate_depth must be an integer >= 1")
        if not np.isfinite(rrf_k) or rrf_k < 0:
            raise ValueError("rrf_k must be finite and >= 0")
        self.encoder = encoder
        self.candidate_depth = candidate_depth
        self.rrf_k = float(rrf_k)
        self.query_instruction = query_instruction
        self._vectors = np.empty((0, encoder.dim), dtype=np.float32)
        self._encoder_fingerprint = encoder.fingerprint()

    def _check_encoder(self) -> None:
        if self.encoder.fingerprint() != self._encoder_fingerprint:
            raise ValueError("encoder configuration changed; rebuild the hybrid retriever")

    def _replace(self, documents: dict[str, Document], changed: list[str], batch_size: int) -> None:
        self._check_encoder()
        corpus = self._build(documents)
        new_vectors = self.encoder.encode([documents[i].text for i in changed], kind="document", batch_size=batch_size)
        old_positions = {id_: i for i, id_ in enumerate(self._corpus.ids)}
        new_positions = {id_: i for i, id_ in enumerate(changed)}
        vectors = np.empty((len(corpus.ids), self.encoder.dim), dtype=np.float32)
        for position, id_ in enumerate(corpus.ids):
            if id_ in new_positions:
                vectors[position] = new_vectors[new_positions[id_]]
            else:
                vectors[position] = self._vectors[old_positions[id_]]
        self._corpus = corpus
        self._vectors = vectors

    def _search_batch(self, queries: list[str], k: int, eligible: np.ndarray) -> list[list[SearchHit]]:
        self._check_encoder()
        query_vectors = self.encoder.encode(queries, kind="query", instruction=self.query_instruction)
        depth = max(k, self.candidate_depth)
        results = []
        for query, vector in zip(queries, query_vectors, strict=True):
            lexical = self._scores(query)
            lexical_ids = self._rank(lexical, eligible[lexical[eligible] > 0], depth)
            dense_ids = self._rank(self._vectors @ vector, eligible, depth)
            fused: dict[int, float] = {}
            for branch in (dense_ids, lexical_ids):
                for rank, position in enumerate(branch, start=1):
                    key = int(position)
                    fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            ranked = sorted(fused, key=lambda position: (-fused[position], position))[:k]
            results.append([self._hit(position, fused[position]) for position in ranked])
        return results

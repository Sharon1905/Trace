"""Deterministic lexical retrieval behind a replaceable retriever interface."""

from __future__ import annotations

import re
from typing import Any, Protocol

from app.embeddings.embedder import source_window_evidence


class Retriever(Protocol):
    def retrieve(self, question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        """Return the most relevant chunks, with scores, for the supplied project."""


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "this", "to", "what",
    "where", "which", "who", "with",
}


def _terms(value: str) -> set[str]:
    terms: set[str] = set()
    for token in _TOKEN.findall(value):
        split_token = _CAMEL_BOUNDARY.sub(" ", token)
        split_token = _DIGIT_BOUNDARY.sub(" ", split_token).replace("_", " ")
        for component in split_token.split():
            normalized = component.casefold()
            if normalized not in _STOP_WORDS and len(normalized) > 1:
                terms.add(normalized)
    return terms


def assess_evidence(question: str, results: list[dict]) -> dict:
    """Require direct source anchors before retrieved code is presented as evidence.

    This is a deterministic grounding check, not a calibrated probability. For
    multi-concept questions, at least two distinct query terms must occur in the
    returned snippets, symbols, or paths; short one- or two-term queries need
    one anchor. Semantic similarity by itself is never treated as proof.
    """
    query_terms = _terms(question)
    required = 1 if len(query_terms) <= 2 else 2
    anchored: set[str] = set()
    for item in results:
        source = " ".join((
            str(item.get("symbol") or ""),
            str(item.get("file_path") or ""),
            str(item.get("code") or ""),
        ))
        anchored.update(query_terms & _terms(source))

    sufficient = len(anchored) >= required
    if sufficient:
        reason = "Retrieved source contains direct query-term anchors."
    elif not results:
        reason = "Retrieval returned no source evidence."
    else:
        reason = (
            f"Retrieved source anchors {len(anchored)} of the query's "
            f"{len(query_terms)} content terms; at least {required} direct "
            "source anchors are required."
        )
    return {
        "status": "supported" if sufficient else "insufficient",
        "sufficient": sufficient,
        "query_content_term_count": len(query_terms),
        "anchored_terms": sorted(anchored),
        "required_anchor_count": required,
        "reason": reason,
    }


class LexicalRetriever:
    """Rank source chunks by query-token coverage; no semantic model is implied."""

    def retrieve(self, question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        if top_k <= 0 or not question.strip():
            return []
        query_terms = _terms(question)
        if not query_terms:
            return []

        matches: list[dict] = []
        for chunk in chunks:
            source = " ".join((
                str(chunk.get("symbol") or ""),
                str(chunk.get("file_path") or ""),
                str(chunk.get("code") or ""),
            ))
            matched_terms = query_terms & _terms(source)
            if not matched_terms:
                continue
            score = len(matched_terms) / len(query_terms)
            matches.append({**chunk, "score": score, "matched_terms": sorted(matched_terms)})

        matches.sort(key=lambda item: (
            -item["score"],
            str(item.get("file_path", "")).casefold(),
            int(item.get("line_start", 0)),
            str(item.get("chunk_id", "")),
        ))
        return matches[:top_k]


class SemanticRetriever:
    """Search a validated project-local FAISS index with query embeddings."""

    def __init__(self, embedder: Any, *, minimum_score: float = 0.30):
        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between -1 and 1")
        self.embedder = embedder
        self.minimum_score = minimum_score

    def retrieve(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 3,
        *,
        search_index: Any | None = None,
    ) -> list[dict]:
        if top_k <= 0 or not question.strip() or search_index is None or search_index.count == 0:
            return []
        vectors = self.embedder.encode_queries([question])
        shape = getattr(vectors, "shape", None)
        invalid_shape = len(vectors) != 1
        if shape is not None:
            invalid_shape = invalid_shape or len(shape) != 2 or int(shape[0]) != 1
            if len(shape) == 2:
                invalid_shape = invalid_shape or int(shape[1]) != int(search_index.dimension)
        elif len(vectors) == 1:
            invalid_shape = len(vectors[0]) != int(search_index.dimension)
        if invalid_shape:
            raise ValueError("Query embedder returned an invalid matrix")
        chunk_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
        retrieved_by_parent: dict[str, dict] = {}
        search_limit = min(search_index.count, max(top_k * 4, top_k))
        for hit in search_index.search(vectors[0], search_limit):
            score = float(hit["score"])
            if score < self.minimum_score:
                continue
            chunk_id = str(hit.get("parent_chunk_id") or hit["chunk_id"])
            source = chunk_by_id.get(chunk_id)
            if source is None:
                raise ValueError("Semantic index returned a chunk absent from the selected project")
            candidate = retrieved_by_parent.get(chunk_id)
            if candidate is not None:
                candidate["matched_window_ids"].append(hit.get("window_id"))
                continue
            evidence = source_window_evidence(source, hit) if hit.get("window_id") else {
                "code": source.get("code", ""),
                "line_start": source.get("line_start", 1),
                "line_end": source.get("line_end", source.get("line_start", 1)),
            }
            retrieved_by_parent[chunk_id] = {
                **source,
                **evidence,
                "score": score,
                "semantic_score": score,
                "semantic_rank": len(retrieved_by_parent) + 1,
                "retrieval_sources": ["semantic"],
                "window_id": hit.get("window_id"),
                "matched_window_ids": [hit.get("window_id")] if hit.get("window_id") else [],
            }
        return list(retrieved_by_parent.values())[:top_k]


class HybridRetriever:
    """Fuse lexical and semantic rankings with reciprocal rank fusion (RRF)."""

    def __init__(self, *, lexical: LexicalRetriever | None = None, rrf_k: int = 60):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self.lexical = lexical or LexicalRetriever()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 3,
        *,
        semantic_results: list[dict] | None = None,
    ) -> list[dict]:
        lexical_results = self.lexical.retrieve(question, chunks, top_k=top_k)
        return self.fuse(lexical_results, semantic_results or [], top_k=top_k)

    def fuse(
        self,
        lexical_results: list[dict],
        semantic_results: list[dict],
        *,
        top_k: int = 3,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        fused: dict[str, dict] = {}
        for source_name, results in (("lexical", lexical_results), ("semantic", semantic_results)):
            for rank, item in enumerate(results, 1):
                chunk_id = str(item.get("chunk_id") or "")
                if not chunk_id:
                    # Historical or caller-provided chunks without IDs are kept
                    # distinct by their source location and contents.
                    chunk_id = "\0".join((
                        str(item.get("file_path", "")),
                        str(item.get("line_start", "")),
                        str(item.get("code", "")),
                    ))
                candidate = fused.setdefault(chunk_id, {
                    **item,
                    "score": 0.0,
                    "retrieval_sources": [],
                })
                candidate["score"] += 1.0 / (self.rrf_k + rank)
                if source_name not in candidate["retrieval_sources"]:
                    candidate["retrieval_sources"].append(source_name)
                if source_name == "lexical":
                    candidate["lexical_score"] = float(item.get("score", 0.0))
                    candidate["lexical_rank"] = rank
                    candidate["matched_terms"] = item.get("matched_terms", [])
                else:
                    candidate["semantic_score"] = float(item.get("semantic_score", item.get("score", 0.0)))
                    candidate["semantic_rank"] = rank
                # Prefer complete chunk metadata if the first result was enriched
                # by a retriever that omitted any optional fields.
                for field in (
                    "file_path", "kind", "symbol", "line_start", "line_end", "part", "code", "window_id",
                ):
                    if field in item:
                        candidate[field] = item[field]

        ordered = sorted(fused.values(), key=lambda item: (
            -item["score"],
            str(item.get("file_path", "")).casefold(),
            int(item.get("line_start", 0)),
            str(item.get("chunk_id", "")),
        ))
        return ordered[:top_k]

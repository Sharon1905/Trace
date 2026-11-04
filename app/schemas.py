"""HTTP request and response schemas for the TRACE API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ID_PATTERN = r"^[0-9a-f]{32}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    project_id: str = Field(min_length=32, max_length=32, pattern=PROJECT_ID_PATTERN)
    question: str = Field(min_length=1, max_length=4000)
    retrieval_mode: Literal["lexical", "semantic", "hybrid"] | None = None

    @field_validator("question")
    @classmethod
    def strip_and_require_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value


class HealthResponse(BaseModel):
    status: str
    version: str


class ParseError(BaseModel):
    file: str
    error: str


class UploadResponse(BaseModel):
    project_id: str
    indexed_chunks: int
    python_files: int
    parse_errors: list[ParseError]
    semantic_index_status: str = "unavailable"
    embedding_model: str | None = None
    semantic_index_error: str | None = None


class SourceReference(BaseModel):
    project_id: str | None = None
    chunk_id: str | None = None
    window_id: str | None = None
    file_path: str
    kind: str | None = None
    line_start: int
    line_end: int
    code: str
    symbol: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    project_id: str
    mode: Literal["repo", "inventory", "file", "function", "usage", "rag"]
    answer: str | None = None
    references: list[SourceReference] = Field(default_factory=list)
    function: str | None = None
    file: str | None = None
    line: int | None = None
    ambiguous: bool | None = None
    usages: list[dict] | None = None
    limitations: str | None = None
    retrieval_mode: str | None = None
    retrieval_diagnostics: dict | None = None
    llm_status: str | None = None  # "configured", "unconfigured", "unavailable"
    debug_data: DebugRetrievalResponse | None = None  # Only populated if debug=true


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    indexed_chunks: int
    python_files: int
    total_files: int = 0
    parse_errors: list[ParseError] = Field(default_factory=list)
    semantic_index_status: str = "unavailable"
    embedding_model: str | None = None
    semantic_index_error: str | None = None
    created_at: str | None = None
    added_at: str | None = None


class ProjectRenameRequest(StrictModel):
    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def strip_and_require_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Project name must not be blank")
        return value


class ProjectDeleteResponse(BaseModel):
    project_id: str
    deleted: bool
    message: str


class ProjectFileInfo(BaseModel):
    path: str
    name: str
    size_bytes: int
    line_count: int
    is_python: bool
    chunks: int = 0
    symbols: list[str] = Field(default_factory=list)
    syntax_error: str | None = None


class ProjectFilesResponse(BaseModel):
    project_id: str
    total_files: int
    python_files: int
    files: list[ProjectFileInfo]


class DebugChunkScore(BaseModel):
    """Individual chunk with full retrieval scoring breakdown for debug mode."""
    chunk_id: str
    file_path: str
    symbol: str | None = None
    kind: str | None = None
    line_start: int
    line_end: int
    code: str

    # Retrieval scores
    lexical_score: float | None = None  # Token coverage (0-1)
    lexical_rank: int | None = None  # Pre-fusion rank
    semantic_score: float | None = None  # Cosine similarity (0-1)
    semantic_rank: int | None = None  # Pre-fusion rank
    hybrid_score: float | None = None  # RRF fused score
    final_rank: int  # Position in result list
    retrieval_sources: list[str]  # ["lexical"] or ["semantic"] or both

    # Evidence gating diagnostics
    matched_query_terms: list[str]  # Query anchors found in this chunk
    evidence_gate_passed: bool  # Did this chunk contribute to passing gate?

    # Additional metadata
    window_id: str | None = None  # Semantic embedding window ID if applicable
    matched_window_ids: list[str] | None = None  # Multiple windows if overlapping matches


class DebugEvidenceSummary(BaseModel):
    """Evidence gating summary for operator understanding."""
    gate_passed: bool  # Did evidence pass the gate?
    sufficient: bool  # Same as gate_passed
    query_term_count: int  # Distinct content terms in question
    required_anchor_count: int  # 1 for short queries, 2 for multi-concept
    anchored_terms: list[str]  # Terms found in source
    gate_reason: str  # Explanation of gate decision
    chunks_passing_gate: int  # How many chunks contributed anchors


class DebugTimingInfo(BaseModel):
    """Performance timing for debug queries."""
    total: float  # End-to-end time in milliseconds
    retrieval: float  # Lexical + semantic retrieval time
    embedding: float | None = None  # Semantic embedding time (if applicable)
    evidence_assessment: float  # Evidence gating assessment time


class DebugRetrievalResponse(BaseModel):
    """Debug mode response with full retrieval metadata, no LLM generation."""
    project_id: str
    question: str
    retrieval_mode: str  # "lexical", "semantic", or "hybrid"
    debug_mode: bool = True

    # Raw retrieval metadata
    raw_results: list[DebugChunkScore]
    result_count: int

    # Retrieval diagnostics (existing data, exposed for operators)
    retrieval_diagnostics: dict = Field(description="Mode, fallback reason, candidate counts, evidence check")

    # Evidence summary for operator
    evidence_summary: DebugEvidenceSummary

    # Timing and performance
    timing_ms: DebugTimingInfo

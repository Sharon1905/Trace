"""FastAPI entrypoint for the reconstructed TRACE baseline."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.analysis.call_graph import find_function_calls
from app.analysis.file_explainer import get_file_content
from app.analysis.function_explainer import get_function_code
from app.analysis.repo_explainer import build_repo_explanation
from app.analysis.repo_summary import summarize_repository
from app.analysis.unused_functions import find_unused_functions
from app.caching.query_cache import QueryCache, get_query_cache, set_query_cache
from app.chunking.chunker import build_code_chunks
from app.embeddings.embedder import EmbeddingError, EmbeddingSettings, SentenceTransformerEmbedder
from app.generation.llm_engine import LLMEngine
from app.generation.prompt_builder import build_prompt
from app.ingestion.extractor import MAX_ARCHIVE_BYTES, ExtractionError, extract_project
from app.monitoring import (
    FileWatcher,
    MetricsCollector,
    RebuildCoordinator,
    RebuildStatus,
    RebuildTrigger,
    WebhookDispatcher,
    get_metrics_collector,
    get_rebuild_coordinator,
    get_webhook_dispatcher,
    set_metrics_collector,
    set_rebuild_coordinator,
    set_webhook_dispatcher,
)
from app.monitoring.build_tracer import BuildTracer, get_build_tracer, set_build_tracer
from app.parsing.ast_parser import parse_python_file
from app.retrieval.retriever import HybridRetriever, LexicalRetriever, SemanticRetriever, assess_evidence
from app.schemas import (
    AskRequest,
    AskResponse,
    DebugChunkScore,
    DebugEvidenceSummary,
    DebugRetrievalResponse,
    DebugTimingInfo,
    HealthResponse,
    ProjectFileInfo,
    ProjectFilesResponse,
    ProjectRenameRequest,
    ProjectDeleteResponse,
    ProjectSummary,
    UploadResponse,
)
from app.storage.project_store import ProjectIndexError, ProjectIndexMissing, ProjectStore
from app.utils.project_manager import get_data_dir, get_project, list_projects
from app.vectorstore.faiss_store import (
    FAISSStore,
    canonical_chunks,
)


class AppState:
    """Application state holder."""

    file_watcher: Optional[FileWatcher] = None
    rebuild_coordinator: Optional[RebuildCoordinator] = None
    query_cache: Optional["QueryCache"] = None
    webhook_dispatcher: Optional[WebhookDispatcher] = None
    metrics_collector: Optional[MetricsCollector] = None


app_state = AppState()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan: startup and shutdown."""
    # Startup
    data_dir = get_data_dir()
    projects_dir = data_dir / "projects"

    # File watcher
    watcher = FileWatcher(projects_dir)
    watcher.start()
    app_state.file_watcher = watcher

    # Rebuild coordinator
    rebuild_coordinator = RebuildCoordinator(watcher=watcher)
    rebuild_coordinator.start()
    app_state.rebuild_coordinator = rebuild_coordinator
    set_rebuild_coordinator(rebuild_coordinator)

    # Query cache
    query_cache = QueryCache(max_entries=1000, default_ttl=3600)
    app_state.query_cache = query_cache
    set_query_cache(query_cache)

    # Webhook dispatcher
    webhook_dispatcher = WebhookDispatcher()
    app_state.webhook_dispatcher = webhook_dispatcher
    set_webhook_dispatcher(webhook_dispatcher)
    rebuild_coordinator.webhook_dispatcher = webhook_dispatcher.dispatch

    # Build tracer
    build_tracer = BuildTracer()
    app_state.build_tracer = build_tracer
    set_build_tracer(build_tracer)

    # Metrics collector
    metrics_collector = MetricsCollector()
    app_state.metrics_collector = metrics_collector
    set_metrics_collector(metrics_collector)

    # Wire rebuild callback to metrics
    def on_rebuild_complete(job):
        metrics_collector.record_rebuild(RebuildMetric(
            job_id=job.job_id,
            project_id=job.project_id,
            trigger=job.trigger.value,
            status=job.status.value,
            duration_ms=(job.completed_at - job.started_at).total_seconds() * 1000
            if job.started_at and job.completed_at else 0,
            chunks_before=0,  # Would need to track
            chunks_after=0,
            incremental=job.incremental,
            changed_files_count=len(job.changed_files),
            error=job.error,
        ))
        # Invalidate cache for this project
        query_cache.invalidate_project(job.project_id)

    rebuild_coordinator.rebuild_callback = on_rebuild_complete

    # Wire file watcher to trigger rebuilds
    def on_file_changes(changes):
        if changes:
            for change in changes:
                project_id = change.get("project_id")
                if project_id and get_project(project_id):
                    changed_file = change.get("file")
                    rebuild_coordinator.queue_rebuild(
                        project_id,
                        trigger=RebuildTrigger.FILE_WATCHER,
                        changed_files=[changed_file] if changed_file else [],
                        incremental=True,
                    )

    watcher.on_change = on_file_changes

    yield  # App runs here

    # Shutdown
    rebuild_coordinator.stop()
    query_cache.shutdown()
    watcher.stop()


app = FastAPI(
    title="TRACE - Code Intelligence Assistant",
    version="0.2.0",
    lifespan=lifespan,
)
_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "TRACE_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Accept", "Content-Type"],
)
llm = LLMEngine()
project_store = ProjectStore()
retriever = LexicalRetriever()
hybrid_retriever = HybridRetriever(lexical=retriever)
semantic_store = FAISSStore()


def _select_project(project_id: str) -> Path:
    project_path = get_project(project_id)
    if project_path is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_path


def _load_project_index(project_path: Path) -> tuple[dict, list[dict]]:
    try:
        metadata, chunks = project_store.load(project_path)
    except ProjectIndexMissing as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProjectIndexError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return metadata, chunks


def _generate(prompt: str, chunks: list[dict]) -> str:
    try:
        return llm.generate(prompt, source_chunks=chunks)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="Configured LLM request failed") from exc


def _reference(item: dict, *, include_score: bool = False) -> dict:
    reference = {
        "chunk_id": item.get("chunk_id"),
        "file_path": item["file_path"],
        "kind": item.get("kind"),
        "line_start": int(item["line_start"]),
        "line_end": int(item["line_end"]),
        "code": str(item.get("code", "")),
        "symbol": item.get("symbol"),
    }
    if item.get("window_id"):
        reference["window_id"] = item["window_id"]
    if item.get("project_id"):
        reference["project_id"] = item["project_id"]
    if include_score:
        reference["score"] = float(item["score"])
    return reference


def _chunk_embedding_text(chunk: dict) -> str:
    return "\n".join(filter(None, (
        f"File: {chunk.get('file_path', '')}",
        f"Type: {chunk.get('kind', '')}",
        f"Symbol: {chunk.get('symbol') or ''}",
        str(chunk.get("code", "")),
    )))


def _build_file_inventory(project_path: Path, chunks: list[dict], metadata) -> dict:
    """Build a useful file inventory for the project.

    Groups files by type: Python implementation, tests, documentation, configuration.
    Identifies key components and classifies each file's purpose.
    """
    from pathlib import Path as PathLib

    root = project_path.resolve()
    files_dict: dict[str, list[dict]] = {
        "Python modules": [],
        "Tests": [],
        "Documentation": [],
        "Configuration": [],
        "Other": [],
    }

    # Categorize files
    for chunk in chunks:
        file_path = PathLib(chunk.get("file_path", ""))
        if not file_path.is_absolute():
            file_path = root / file_path

        # Skip hidden files, build artifacts, and cache directories
        if file_path.name.startswith("."):
            continue
        if ".trace" in file_path.parts:
            continue
        if file_path.parent.name in {".git", "node_modules", "__pycache__", ".venv", ".venv-verify", "dist", "build"}:
            continue

        # Determine category
        name = chunk.get("symbol", "") or file_path.stem
        is_test_file = file_path.match("test_*.py") or file_path.match("*_test.py") or any(part.startswith("test") for part in file_path.parts)
        if is_test_file:
            category = "Tests"
        elif file_path.suffix == ".py":
            name = chunk.get("symbol", "")
            if not name:  # Go by definitions in file
                for definition in chunk.get("definitions", []) or []:
                    name = definition.get("qualified_name", "")
                if not name:
                    # Extract from filename if no symbols found
                    name = file_path.stem

            category = "Python modules"
        elif file_path.suffix.lower() in {".md", ".rst", ".txt"}:
            category = "Documentation"
        elif any(file_path.match(ext) for ext in ["*.yaml", "*.yml", "*.json", "*.toml", "*.cfg", "*.ini", "*.env*"]):
            category = "Configuration"
        else:
            category = "Other"

        files_dict[category].append({
            "path": str(file_path.relative_to(root)),
            "name": file_path.name,
            "category": category,
            "symbol": name,
            "line_count": chunk.get("line_end", 0) - chunk.get("line_start", 0) + 1,
        })

    # Build inventory response
    lines = ["# File Inventory for this Project\n\n"]
    lines.append(f"**Project Overview:**\n")
    lines.append(f"- **Total Files:** {len(chunks)} indexed files\n")
    lines.append(f"- **Python Files:** {len(files_dict['Python modules'])}\n")
    lines.append(f"- **Tests:** {len(files_dict['Tests'])}\n")
    lines.append(f"- **Documentation:** {len(files_dict['Documentation'])}\n\n")

    lines.append("**Key Components:**\n")
    python_files = files_dict["Python modules"]
    for item in python_files[:20]:  # Limit to reasonable number
        symbol = item["symbol"] or "module"
        if "(" in symbol:
            symbol = symbol.split("(")[0]
        lines.append(f"- **{symbol}** in [{item['path']}]")

    if len(python_files) > 20:
        lines.append(f"\n*... and {len(python_files) - 20} more Python modules. *\n")

    lines.append("\n**File Categories:**\n")
    for category, items in files_dict.items():
        if items:
            lines.append(f"\n## {category} ({len(items)})\n")
            for item in items[:10]:  # Show first 10 per category
                symbol = item["symbol"] or "unnamed"
                if "(" in symbol:
                    symbol = symbol.split("(")[0]
                lines.append(f"- **[{item['path']}]** - {symbol}")
            if len(items) > 10:
                lines.append(f"\n  *... and {len(items) - 10} more {category.lower()} files.*")

    return {
        "project_id": metadata.get("project_id", ""),
        "mode": "inventory",
        "answer": "\n".join(lines),
        "references": [],
    }


def _build_repository_context(project_path: Path, metadata, read_important_files: bool = True) -> dict:
    """Build a rich repository context summary with project metadata.

    Uses file structure, README, manifest, Python docstrings, and important files.
    Works even when README is missing by using metadata, structure, and code.
    """
    from pathlib import Path as PathLib

    root = project_path.resolve()

    lines = []
    lines.append("# Project Context\n\n")

    # Project identification
    lines.append(f"**Project ID:** `{metadata.get('project_id', 'unknown')}`\n")
    lines.append(f"**Created:** {metadata.get('created_at', 'unknown')}\n")
    lines.append(f"**File Count:** {metadata.get('file_count', 'unknown')} total files\n")
    lines.append(f"**Python Files:** {metadata.get('python_file_count', 'unknown')}\n")
    lines.append(f"**Indexed Chunks:** {metadata.get('indexed_chunk_count', 'unknown')}\n\n")

    # Detection retrieval backend status
    retrieval_backend = metadata.get('retrieval_backend', 'unknown')
    lines.append(f"**Retrieval Backend:** {retrieval_backend}\n\n")

    # Check for README
    readme_path = root / "README.md"
    if readme_path.exists():
        lines.append("## Project Description (from README)\n\n")
        try:
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                readme_content = f.read(4096)  # Limit to avoid overwhelming
                lines.append(readme_content[:500] + "\n...")
        except Exception as e:
            lines.append(f"*(Could not read README: {e})*\n")

        lines.append("\n---\n\n")

    # Project structure overview
    lines.append("## Project Structure\n\n")

    # Get top-level directories
    try:
        dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if dirs:
            lines.append("**Directory Structure:**\n")
            for d in sorted(dirs, key=lambda x: x.name):
                # Count files in directory
                file_count = sum(1 for f in d.rglob("*") if f.is_file() and ".trace" not in f.relative_to(root))
                lines.append(f"- **`{d.name}`** ({file_count} files)")

        lines.append("\n---\n\n")
    except PermissionError:
        pass

    # Important Python files and their purposes
    lines.append("## Key Modules and Components\n\n")

    # Group meaningful Python files by pattern
    python_files = [root / f.relative_to(root) for f in root.rglob("*.py") if ".trace" not in f.relative_to(root).parts]

    # Entrypoints
    entrypoints = [f for f in python_files if f.name in {"main.py", "app.py", "__main__.py"}]
    if entrypoints:
        lines.append("### Entry Points\n")
        for f in entrypoints:
            lines.append(f"- **`{f.relative_to(root)}`** - Main application entry point\n")
        lines.append("\n")

    # Configuration files
    config_files = [f for f in python_files if
        f.name in {"config.py", "settings.py", "conftest.py"} or
        (".config" in f.name or "conftest" in f.name)]
    if config_files:
        lines.append("### Configuration\n")
        for f in config_files:
            lines.append(f"- **`{f.relative_to(root)}`** - Configuration or fixture setup\n")
        lines.append("\n")

    # Other important modules - use symbols from indexed chunks
    from app.parsing.ast_parser import parse_python_file

    meaningful_files = {}
    for chunk in metadata.get("chunks", []):
        file_path = chunk.get("file_path", "")
        if not file_path:
            continue

        full_path = root / file_path
        if not full_path.exists() or not full_path.is_file():
            continue

        symbol = chunk.get("symbol", "")
        if not symbol:
            # Try to get definitions
            parsed = parse_python_file(full_path, root)
            definitions = [d.get("qualified_name", "") for d in parsed.get("definitions", [])]
            if definitions:
                symbol = definitions[0]
            else:
                continue

        file_key = full_path.relative_to(root)
        if file_key not in meaningful_files:
            meaningful_files[file_key] = []
        meaningful_files[file_key].append(symbol)

    # Group by filename for display
    if meaningful_files:
        lines.append("### Application Modules\n")
        # Prioritize files with multiple symbols or well-known patterns
        sorted_files = sorted(
            meaningful_files.keys(),
            key=lambda x: (-len(meaningful_files[x]), x.name.casefold())
        )

        for file_path in sorted_files[:30]:  # Limit
            symbols = meaningful_files[file_path]
            # Take first symbol as main identifier
            main_symbol = symbols[0] if symbols else "module"
            if "(" in main_symbol:
                main_symbol = main_symbol.split("(")[0]

            # Provide brief description based on symbol name
            desc = _describe_symbol(main_symbol)
            lines.append(f"- **[{file_path}]** - {desc}")
            if len(symbols) > 1:
                lines.append(f"  Defines: {', '.join([s.split('(')[0] for s in symbols])}")

        lines.append("\n")

    # Tests
    test_files = [f for f in python_files if "test_" in f.name or "conftest" in f.name or any(part.startswith("test") for part in f.parts)]
    if test_files:
        lines.append("## Tests\n\n")
        lines.append(f"{len(test_files)} test files found, providing test coverage for:\n")
        for f in test_files[:10]:
            lines.append(f"- **`{f.relative_to(root)}`**")
        if len(test_files) > 10:
            lines.append(f"\n*... and {len(test_files) - 10} more test files.*\n")

    lines.append("\n---\n\n")

    # Disclaimer
    lines.append("## Limitations\n\n")
    lines.append(f"- Extracted from {metadata.get('indexed_chunk_count', 'unknown')} indexed chunks\n")
    lines.append("- Some files may not be fully explicated\n")
    lines.append("- Test files included in inventory for batch coverage visibility\n")
    lines.append(f"- Project ID: `{metadata.get('project_id', '')}`\n")

    return {
        "project_id": metadata.get("project_id", ""),
        "mode": "repo",
        "answer": "\n".join(lines),
        "references": [],
        "metadata": metadata,
    }


def _describe_symbol(symbol: str) -> str:
    """Provide a natural-language description from a symbol name string."""
    if not symbol:
        return "module"
    lowered = symbol.casefold()
    # Common naming conventions
    if any(word in lowered for word in ("test_", "_test")):
        return f"Test: {symbol}"
    if any(word in lowered for word in ("config", "setting", "option")):
        return f"Configuration: {symbol}"
    if any(word in lowered for word in ("util", "helper", "tool")):
        return f"Utility: {symbol}"
    if any(word in lowered for word in ("model", "schema", "entity")):
        return f"Data model: {symbol}"
    if any(word in lowered for word in ("route", "view", "handler", "endpoint")):
        return f"Endpoint: {symbol}"
    if any(word in lowered for word in ("service", "manager", "controller")):
        return f"Service: {symbol}"
    if any(word in lowered for word in ("db", "database", "store", "repository")):
        return f"Storage: {symbol}"
    if any(word in lowered for word in ("client", "connect", "request")):
        return f"Client: {symbol}"
    if any(word in lowered for word in ("parse", "serial", "format", "convert")):
        return f"Processing: {symbol}"
    if symbol[0].isupper():
        return f"Class: {symbol}"
    return symbol


def _build_semantic_index(
    project_path: Path,
    project_id: str,
    chunks: list[dict],
    settings: EmbeddingSettings | None = None,
) -> dict:
    """Build one project's index and return its manifest for project metadata."""
    import numpy as np

    settings = settings or EmbeddingSettings.from_env()
    ordered = canonical_chunks(chunks)
    embedder = SentenceTransformerEmbedder(settings)
    if ordered:
        vectors, windows = embedder.encode_chunk_documents(ordered)
    else:
        vectors = np.empty((0, 0), dtype=np.float32)
        windows = []
    embedding_config = embedder.index_config
    header = semantic_store.save(
        project_path,
        project_id,
        ordered,
        vectors,
        model_name=settings.model_name,
        embedding_config=embedding_config,
        vector_mapping=windows,
    )
    return {
        "status": "ready",
        "model_name": settings.model_name,
        "embedding_config": embedding_config,
        "dimension": header["dimension"],
        "chunk_count": len(ordered),
        "window_count": header["count"],
        "corpus_fingerprint": header["corpus_fingerprint"],
        "index_file": FAISSStore.INDEX_NAME,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _unavailable_semantic_index(exc: Exception, settings: EmbeddingSettings | None = None) -> dict:
    return {
        "status": "unavailable",
        "model_name": settings.model_name if settings else None,
        "embedding_config": settings.as_dict() if settings else None,
        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
    }


def _load_or_rebuild_semantic_index(
    project_path: Path,
    metadata: dict,
    chunks: list[dict],
) -> tuple[object | None, EmbeddingSettings | None, str | None]:
    """Validate persisted state and rebuild stale/missing state atomically."""
    settings: EmbeddingSettings | None = None
    try:
        settings = EmbeddingSettings.from_env()
        embedder = SentenceTransformerEmbedder(settings)
        windows = embedder.document_windows(chunks)
        embedding_config = embedder.index_config
        try:
            loaded = semantic_store.load(
                project_path,
                str(metadata["project_id"]),
                chunks,
                model_name=settings.model_name,
                embedding_config=embedding_config,
                vector_mapping=windows,
            )
            return loaded, settings, None
        except Exception as stale_or_missing:
            try:
                import numpy as np

                vectors = embedder.encode_document_windows(windows) if windows else np.empty((0, 0), dtype=np.float32)
                header = semantic_store.save(
                    project_path,
                    str(metadata["project_id"]),
                    chunks,
                    vectors,
                    model_name=settings.model_name,
                    embedding_config=embedding_config,
                    vector_mapping=windows,
                )
                descriptor = {
                    "status": "ready",
                    "model_name": settings.model_name,
                    "embedding_config": embedding_config,
                    "dimension": header["dimension"],
                    "chunk_count": header["chunk_count"],
                    "window_count": header["count"],
                    "corpus_fingerprint": header["corpus_fingerprint"],
                    "index_file": FAISSStore.INDEX_NAME,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                loaded = semantic_store.load(
                    project_path,
                    str(metadata["project_id"]),
                    chunks,
                    model_name=settings.model_name,
                    embedding_config=embedding_config,
                    vector_mapping=windows,
                )
                metadata["semantic_index"] = descriptor
                metadata["retrieval_backend"] = "faiss-cosine+lexical-rrf"
                project_store.save(project_path, metadata, chunks)
                return loaded, settings, None
            except Exception as exc:
                unavailable = _unavailable_semantic_index(exc, settings)
                metadata["semantic_index"] = unavailable
                metadata["retrieval_backend"] = "lexical-token-overlap"
                try:
                    project_store.save(project_path, metadata, chunks)
                except Exception:
                    pass
                reason = f"{stale_or_missing}; rebuild failed: {unavailable['error']}"
                return None, settings, reason
    except Exception as exc:
        unavailable = _unavailable_semantic_index(exc, settings)
        metadata["semantic_index"] = unavailable
        metadata["retrieval_backend"] = "lexical-token-overlap"
        try:
            project_store.save(project_path, metadata, chunks)
        except Exception:
            pass
        return None, settings, unavailable["error"]


def _minimum_semantic_score() -> float:
    raw = os.getenv("TRACE_SEMANTIC_MIN_SCORE", "0.30")
    try:
        score = float(raw)
    except ValueError as exc:
        raise EmbeddingError("TRACE_SEMANTIC_MIN_SCORE must be a number") from exc
    if not -1.0 <= score <= 1.0:
        raise EmbeddingError("TRACE_SEMANTIC_MIN_SCORE must be between -1 and 1")
    return score


def _requested_retrieval_mode(explicit_mode: str | None) -> tuple[str, str | None]:
    if explicit_mode:
        return explicit_mode, None
    configured = os.getenv("TRACE_RETRIEVAL_MODE", "hybrid").strip().casefold()
    if configured in {"lexical", "semantic", "hybrid"}:
        return configured, None
    return "hybrid", f"Unsupported TRACE_RETRIEVAL_MODE {configured!r}; using hybrid"


def _retrieve_rag(
    question: str,
    project_path: Path,
    metadata: dict,
    chunks: list[dict],
    requested_mode: str,
    *,
    configuration_note: str | None = None,
) -> tuple[list[dict], dict]:
    def scoped(results: list[dict]) -> list[dict]:
        return [{**item, "project_id": str(metadata["project_id"])} for item in results]

    candidate_k = 10
    lexical_results = retriever.retrieve(question, chunks, top_k=candidate_k)
    diagnostics: dict = {
        "requested_mode": requested_mode,
        "used_mode": requested_mode,
        "fallback_reason": configuration_note,
        "lexical_candidate_count": len(lexical_results),
        "semantic_candidate_count": 0,
        "metric": "lexical-token-coverage",
    }
    if requested_mode == "lexical":
        return scoped(lexical_results[:3]), diagnostics

    search_index, settings, index_error = _load_or_rebuild_semantic_index(
        project_path,
        metadata,
        chunks,
    )
    if search_index is None or settings is None:
        diagnostics.update({
            "used_mode": "lexical",
            "fallback_reason": index_error or "Semantic index is unavailable",
        })
        return scoped(lexical_results[:3]), diagnostics

    try:
        minimum_score = _minimum_semantic_score()
        semantic_retriever = SemanticRetriever(
            SentenceTransformerEmbedder(settings),
            minimum_score=minimum_score,
        )
        semantic_results = semantic_retriever.retrieve(
            question,
            chunks,
            top_k=candidate_k,
            search_index=search_index,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {str(exc)[:500]}"
        diagnostics.update({
            "used_mode": "lexical",
            "fallback_reason": reason,
            "semantic_model": settings.model_name,
        })
        return scoped(lexical_results[:3]), diagnostics

    diagnostics.update({
        "semantic_candidate_count": len(semantic_results),
        "semantic_model": settings.model_name,
        "semantic_dimension": search_index.dimension,
        "index_chunk_count": int(search_index.header.get("chunk_count", search_index.count)),
        "index_window_count": search_index.count,
        "metric": "cosine_similarity_exact_faiss_flat_ip",
        "minimum_semantic_score": minimum_score,
    })
    if requested_mode == "semantic":
        return scoped(semantic_results[:3]), diagnostics

    fused = hybrid_retriever.fuse(
        lexical_results,
        semantic_results,
        top_k=3,
    )
    diagnostics.update({
        "used_mode": "hybrid",
        "metric": "reciprocal_rank_fusion",
        "fusion_k": hybrid_retriever.rrf_k,
    })
    return scoped(fused), diagnostics


def _source_answer(question: str, mode: str, project_id: str, item: dict) -> dict:
    reference = _reference(item)
    prompt = build_prompt(question, [item])
    return {
        "project_id": project_id,
        "mode": mode,
        "answer": _generate(prompt, [item]),
        "references": [reference],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/projects", response_model=list[ProjectSummary])
def get_all_projects() -> list[dict]:
    project_ids = list_projects()
    summaries: list[dict] = []
    for pid in project_ids:
        project_path = get_project(pid)
        if project_path is None:
            continue
        try:
            metadata, chunks = project_store.load(project_path)
            sem = metadata.get("semantic_index") or {}
            name = metadata.get("name")
            if not name:
                non_trace_dirs = [p.name for p in project_path.iterdir() if p.is_dir() and p.name != ".trace"]
                if non_trace_dirs:
                    name = non_trace_dirs[0].replace("_", " ").title()
                else:
                    name = f"Repository {pid[:8]}"
            summaries.append({
                "project_id": pid,
                "name": name,
                "indexed_chunks": metadata.get("indexed_chunk_count", len(chunks)),
                "python_files": metadata.get("python_file_count", 0),
                "total_files": metadata.get("file_count", 0),
                "parse_errors": metadata.get("parse_errors", []),
                "semantic_index_status": sem.get("status", "unavailable"),
                "embedding_model": sem.get("model_name"),
                "semantic_index_error": sem.get("error"),
                "created_at": metadata.get("created_at"),
                "added_at": metadata.get("created_at"),
            })
        except Exception:
            continue
    summaries.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return summaries


@app.patch("/projects/{project_id}", response_model=ProjectSummary)
def rename_project(project_id: str, request: ProjectRenameRequest) -> dict:
    project_path = _select_project(project_id)
    metadata, chunks = _load_project_index(project_path)
    metadata["name"] = request.name
    project_store.save(project_path, metadata, chunks)

    sem = metadata.get("semantic_index") or {}
    return {
        "project_id": project_id,
        "name": request.name,
        "indexed_chunks": metadata.get("indexed_chunk_count", len(chunks)),
        "python_files": metadata.get("python_file_count", 0),
        "total_files": metadata.get("file_count", 0),
        "parse_errors": metadata.get("parse_errors", []),
        "semantic_index_status": sem.get("status", "unavailable"),
        "embedding_model": sem.get("model_name"),
        "semantic_index_error": sem.get("error"),
        "created_at": metadata.get("created_at"),
        "added_at": metadata.get("created_at"),
    }


def _safe_rmtree(path: Path) -> None:
    """Safely remove a directory tree, clearing Windows read-only attributes if needed."""
    if not path.exists():
        return

    def _handle_remove_readonly(func, p, exc):
        import stat
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    shutil.rmtree(path, onerror=_handle_remove_readonly)
    if path.exists():
        raise RuntimeError(f"Directory {path} could not be completely removed")


@app.delete("/projects/{project_id}", response_model=ProjectDeleteResponse)
def delete_project(project_id: str) -> dict:
    project_path = _select_project(project_id)

    # 1. Cancel queued rebuild jobs in coordinator
    coordinator = app_state.rebuild_coordinator
    if coordinator is not None:
        coordinator.cancel_jobs_for_project(project_id)

    # 2. Cancel watcher timers and pending file changes
    watcher = app_state.file_watcher
    if watcher is not None:
        watcher.cancel_project(project_id)

    # Ensure any job that raced during cancellation is cleaned up
    if coordinator is not None:
        coordinator.cancel_jobs_for_project(project_id)

    # 3. Invalidate project-specific query cache
    cache = app_state.query_cache
    if cache is not None:
        cache.invalidate_project(project_id)

    # 4. Remove project directory (chunks, metadata, FAISS index, source files)
    try:
        _safe_rmtree(project_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete project directory: {exc}") from exc

    return {
        "project_id": project_id,
        "deleted": True,
        "message": f"Project {project_id} deleted successfully",
    }



@app.post("/upload", response_model=UploadResponse)
async def upload_project(file: UploadFile = File(...)) -> dict:
    filename = file.filename or ""
    if Path(filename).suffix.casefold() != ".zip":
        await file.close()
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed")

    data_dir = get_data_dir()
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    project_id: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="trace-", suffix=".zip", dir=upload_dir, delete=False) as stream:
            temporary_path = Path(stream.name)
            total = 0
            while block := await file.read(1024 * 1024):
                total += len(block)
                if total > MAX_ARCHIVE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"ZIP upload exceeds the configured {MAX_ARCHIVE_BYTES}-byte limit",
                    )
                stream.write(block)

        project_id, python_files = extract_project(temporary_path, data_dir)
        project_path = get_project(project_id, data_dir)
        if project_path is None:
            raise HTTPException(status_code=500, detail="Extracted project could not be located")

        chunks: list[dict] = []
        parse_errors: list[dict] = []
        for python_file in python_files:
            parsed = parse_python_file(python_file, project_path)
            chunks.extend(build_code_chunks(parsed))
            if parsed["syntax_error"]:
                parse_errors.append({"file": parsed["file_path"], "error": parsed["syntax_error"]})

        try:
            embedding_settings = EmbeddingSettings.from_env()
        except Exception:
            embedding_settings = None
        try:
            semantic_descriptor = _build_semantic_index(
                project_path,
                project_id,
                chunks,
                embedding_settings,
            )
        except Exception as exc:
            semantic_descriptor = _unavailable_semantic_index(exc, embedding_settings)

        metadata = {
            "project_id": project_id,
            "name": Path(filename).stem.replace("-", " ").replace("_", " ").title() if filename else f"Repository {project_id[:8]}",
            "filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file_count": sum(
                1 for path in project_path.rglob("*")
                if path.is_file() and ".trace" not in path.relative_to(project_path).parts
            ),
            "python_file_count": len(python_files),
            "indexed_chunk_count": len(chunks),
            "parse_errors": parse_errors,
            "retrieval_backend": (
                "faiss-cosine+lexical-rrf"
                if semantic_descriptor["status"] == "ready"
                else "lexical-token-overlap"
            ),
            "semantic_index": semantic_descriptor,
        }
        project_store.save(project_path, metadata, chunks)
        return {
            "project_id": project_id,
            "indexed_chunks": len(chunks),
            "python_files": len(python_files),
            "parse_errors": parse_errors,
            "semantic_index_status": semantic_descriptor["status"],
            "embedding_model": semantic_descriptor.get("model_name"),
            "semantic_index_error": semantic_descriptor.get("error"),
        }
    except ExtractionError as exc:
        if project_id:
            shutil.rmtree(data_dir / "projects" / project_id, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        if project_id:
            shutil.rmtree(data_dir / "projects" / project_id, ignore_errors=True)
        raise
    except Exception as exc:
        if project_id:
            shutil.rmtree(data_dir / "projects" / project_id, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Project ingestion failed") from exc
    finally:
        await file.close()
        if temporary_path:
            temporary_path.unlink(missing_ok=True)


def _build_debug_response(
    project_id: str,
    question: str,
    retrieval_mode: str,
    retrieved: list[dict],
    retrieval_diagnostics: dict,
    timing_ms: dict,
) -> DebugRetrievalResponse:
    """Build a debug response with full retrieval metadata and scoring."""
    evidence_check = retrieval_diagnostics.get("evidence_check", {})

    # Build debug chunk scores with all available metadata
    debug_chunks: list[DebugChunkScore] = []
    for rank, item in enumerate(retrieved, 1):
        # Determine which query terms matched this chunk (for evidence gate)
        matched_terms = item.get("matched_terms", [])
        if not matched_terms:
            # For semantic results, check if any anchors from evidence_check match
            source_text = (
                f"{item.get('symbol', '')} {item['file_path']} {item.get('code', '')}"
            ).lower()
            anchored = evidence_check.get("anchored_terms", [])
            matched_terms = [t for t in anchored if t.lower() in source_text]

        debug_chunk = DebugChunkScore(
            chunk_id=item.get("chunk_id", ""),
            file_path=item["file_path"],
            symbol=item.get("symbol"),
            kind=item.get("kind"),
            line_start=int(item["line_start"]),
            line_end=int(item["line_end"]),
            code=str(item.get("code", "")),
            lexical_score=item.get("lexical_score"),
            lexical_rank=item.get("lexical_rank"),
            semantic_score=item.get("semantic_score"),
            semantic_rank=item.get("semantic_rank"),
            hybrid_score=item.get("score") if "semantic_rank" in item or "lexical_rank" in item else None,
            final_rank=rank,
            retrieval_sources=item.get("retrieval_sources", []),
            matched_query_terms=matched_terms,
            evidence_gate_passed=len(matched_terms) > 0,
            window_id=item.get("window_id"),
            matched_window_ids=item.get("matched_window_ids"),
        )
        debug_chunks.append(debug_chunk)

    # Build evidence summary
    evidence_summary = DebugEvidenceSummary(
        gate_passed=evidence_check.get("sufficient", False),
        sufficient=evidence_check.get("sufficient", False),
        query_term_count=evidence_check.get("query_content_term_count", 0),
        required_anchor_count=evidence_check.get("required_anchor_count", 0),
        anchored_terms=evidence_check.get("anchored_terms", []),
        gate_reason=evidence_check.get("reason", "No evidence check performed"),
        chunks_passing_gate=sum(1 for c in debug_chunks if c.evidence_gate_passed),
    )

    return DebugRetrievalResponse(
        project_id=project_id,
        question=question,
        retrieval_mode=retrieval_mode,
        debug_mode=True,
        raw_results=debug_chunks,
        result_count=len(debug_chunks),
        retrieval_diagnostics=retrieval_diagnostics,
        evidence_summary=evidence_summary,
        timing_ms=DebugTimingInfo(**timing_ms),
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    debug: bool = Query(False, description="Return retrieval metadata without LLM generation"),
) -> dict:
    project_path = _select_project(request.project_id)
    question = request.question
    from app.core.router import detect_intent

    intent = detect_intent(question)

    # Debug mode is only for RAG queries
    if debug and intent["type"] != "rag":
        raise HTTPException(status_code=400, detail="Debug mode is only supported for general questions (RAG mode)")

    if intent["type"] == "repo":
        summary = summarize_repository(project_path)
        return {
            "project_id": request.project_id,
            "mode": "repo",
            "answer": build_repo_explanation(summary),
            "references": [],
        }
    if intent["type"] == "file":
        relative_path, code = get_file_content(project_path, intent["target"])
        if code is None:
            raise HTTPException(status_code=404, detail="File not found")
        snippet = code[:16000]
        lines = snippet.splitlines()
        item = {
            "file_path": relative_path,
            "line_start": 1,
            "line_end": max(1, len(lines)),
            "code": snippet,
            "symbol": None,
        }
        return _source_answer(question, "file", request.project_id, item)
    if intent["type"] == "function":
        result = get_function_code(project_path, intent["target"])
        if not result:
            raise HTTPException(status_code=404, detail="Function not found")
        snippet = result["code"][:16000]
        lines = snippet.splitlines()
        item = {
            "file_path": result["file"],
            "line_start": result["line"],
            "line_end": result["line"] + max(0, len(lines) - 1),
            "code": snippet,
            "symbol": result["qualified_name"],
        }
        response = _source_answer(question, "function", request.project_id, item)
        response.update({
            "function": result["qualified_name"],
            "file": result["file"],
            "line": result["line"],
            "ambiguous": result["ambiguous"],
        })
        return response
    if intent["type"] == "usage":
        return {
            "project_id": request.project_id,
            "mode": "usage",
            "function": intent["target"],
            "usages": find_function_calls(project_path, intent["target"]),
            "limitations": "Static call-site search can miss dynamic or reflective calls.",
            "references": [],
        }
    if intent["type"] == "inventory":
        metadata, chunks = _load_project_index(project_path)
        return _build_file_inventory(project_path, chunks, metadata)

    metadata, chunks = _load_project_index(project_path)
    requested_mode, configuration_note = _requested_retrieval_mode(request.retrieval_mode)

    # Track timing for debug mode
    t0_total = time.perf_counter()
    t0_retrieval = time.perf_counter()
    retrieved, retrieval_diagnostics = _retrieve_rag(
        question,
        project_path,
        metadata,
        chunks,
        requested_mode,
        configuration_note=configuration_note,
    )
    t1_retrieval = time.perf_counter()

    t0_evidence = time.perf_counter()
    evidence_check = assess_evidence(question, retrieved)
    t1_evidence = time.perf_counter()
    retrieval_diagnostics["evidence_check"] = evidence_check

    if not evidence_check["sufficient"]:
        retrieved = []

    # DEBUG MODE: Return metadata without LLM generation
    if debug:
        t1_total = time.perf_counter()
        timing_ms = {
            "total": (t1_total - t0_total) * 1000,
            "retrieval": (t1_retrieval - t0_retrieval) * 1000,
            "evidence_assessment": (t1_evidence - t0_evidence) * 1000,
            "embedding": retrieval_diagnostics.get("embedding_time_ms"),
        }
        debug_response = _build_debug_response(
            request.project_id,
            question,
            retrieval_diagnostics["used_mode"],
            retrieved,
            retrieval_diagnostics,
            timing_ms,
        )
        return {
            "project_id": request.project_id,
            "mode": "rag",
            "answer": None,
            "references": [],
            "retrieval_mode": retrieval_diagnostics["used_mode"],
            "retrieval_diagnostics": retrieval_diagnostics,
            "debug_data": debug_response,
        }

    if not retrieved:
        answer = "I do not have enough directly matching source evidence to answer that from this project."
        return {
            "project_id": request.project_id,
            "mode": "rag",
            "answer": answer,
            "references": [],
            "retrieval_mode": retrieval_diagnostics["used_mode"],
            "retrieval_diagnostics": retrieval_diagnostics,
        }

    prompt = build_prompt(question, retrieved)
    return {
        "project_id": request.project_id,
        "mode": "rag",
        "answer": _generate(prompt, retrieved),
        "references": [_reference(item, include_score=True) for item in retrieved],
        "retrieval_mode": retrieval_diagnostics["used_mode"],
        "retrieval_diagnostics": retrieval_diagnostics,
    }


@app.get("/function_usage")
def function_usage(
    function_name: str = Query(..., min_length=1, max_length=256),
    project_id: str = Query(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
) -> dict:
    if not function_name.strip():
        raise HTTPException(status_code=422, detail="function_name must not be blank")
    project_path = _select_project(project_id)
    return {
        "project_id": project_id,
        "function": function_name,
        "usages": find_function_calls(project_path, function_name),
        "limitations": "Static call-site search can miss dynamic or reflective calls.",
    }


@app.get("/unused_functions")
def unused_functions(
    project_id: str = Query(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
) -> dict:
    project_path = _select_project(project_id)
    return {
        "project_id": project_id,
        "unused_functions": find_unused_functions(project_path),
        "limitations": "Candidates are not proof of dead code; dynamic calls and framework conventions may be invisible.",
    }


@app.get("/explain_repo")
def explain_repo(
    project_id: str = Query(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
) -> dict:
    project_path = _select_project(project_id)
    summary = summarize_repository(project_path)
    return {"project_id": project_id, "summary": summary.splitlines(), "explanation": build_repo_explanation(summary)}


@app.get("/project_files", response_model=ProjectFilesResponse)
def get_project_files(
    project_id: str = Query(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
) -> dict:
    project_path = _select_project(project_id)
    metadata, chunks = _load_project_index(project_path)

    chunks_by_file: dict[str, list[dict]] = {}
    for c in chunks:
        fp = c.get("file_path", "")
        chunks_by_file.setdefault(fp, []).append(c)

    parse_errors_by_file = {
        err.get("file"): err.get("error") for err in metadata.get("parse_errors", []) if isinstance(err, dict)
    }

    files_list = []
    python_count = 0
    for path in sorted(project_path.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(project_path).parts
        if ".trace" in rel_parts or any(p in {".git", "__pycache__", "node_modules", ".venv", ".venv-verify"} for p in rel_parts):
            continue
        rel_str = str(path.relative_to(project_path)).replace("\\", "/")
        is_py = path.suffix.casefold() == ".py"
        if is_py:
            python_count += 1

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = sum(1 for _ in f)
        except Exception:
            lines = 0

        file_chunks = chunks_by_file.get(rel_str, [])
        symbols: list[str] = []
        for fc in file_chunks:
            sym = fc.get("symbol")
            if sym and sym not in symbols:
                symbols.append(sym)

        files_list.append({
            "path": rel_str,
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "line_count": lines,
            "is_python": is_py,
            "chunks": len(file_chunks),
            "symbols": symbols,
            "syntax_error": parse_errors_by_file.get(rel_str),
        })

    return {
        "project_id": project_id,
        "total_files": len(files_list),
        "python_files": python_count,
        "files": files_list,
    }



@app.get("/explain_file")
def explain_file(
    filename: str = Query(..., min_length=1, max_length=1024),
    project_id: str = Query(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
) -> dict:
    if not filename.strip():
        raise HTTPException(status_code=422, detail="filename must not be blank")
    project_path = _select_project(project_id)
    relative_path, code = get_file_content(project_path, filename)
    if code is None:
        raise HTTPException(status_code=404, detail="File not found")
    snippet = code[:16000]
    lines = snippet.splitlines()
    item = {
        "file_path": relative_path,
        "line_start": 1,
        "line_end": max(1, len(lines)),
        "code": snippet,
        "symbol": None,
    }
    prompt = build_prompt(f"Explain the file {relative_path}.", [item])
    return {
        "project_id": project_id,
        "file": relative_path,
        "explanation": _generate(prompt, [item]),
        "reference": _reference(item),
    }


@app.get("/explain_function")
def explain_function(
    function_name: str = Query(..., min_length=1, max_length=256),
    project_id: str = Query(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
) -> dict:
    if not function_name.strip():
        raise HTTPException(status_code=422, detail="function_name must not be blank")
    project_path = _select_project(project_id)
    result = get_function_code(project_path, function_name)
    if not result:
        raise HTTPException(status_code=404, detail="Function not found")
    snippet = result["code"][:16000]
    lines = snippet.splitlines()
    item = {
        "file_path": result["file"],
        "line_start": result["line"],
        "line_end": result["line"] + max(0, len(lines) - 1),
        "code": snippet,
        "symbol": result["qualified_name"],
    }
    prompt = build_prompt(f"Explain function {result['qualified_name']}.", [item])
    return {
        "project_id": project_id,
        "function": result["qualified_name"],
        "file": result["file"],
        "line": result["line"],
        "ambiguous": result["ambiguous"],
        "explanation": _generate(prompt, [item]),
        "reference": _reference(item),
    }


@app.get("/monitor/status")
def monitor_status() -> dict:
    """Debug endpoint: Return file watcher status."""
    if not app_state.file_watcher:
        return {"status": "disabled", "reason": "no watcher initialized"}
    return app_state.file_watcher.status()


@app.get("/monitor/changes")
def monitor_changes() -> dict:
    """Debug endpoint: Return tracked file changes since last call."""
    if not app_state.file_watcher:
        return {"changes": []}
    changes = app_state.file_watcher.get_changes()
    return {"changes": changes, "count": len(changes)}


# Rebuild Coordinator endpoints
@app.get("/monitor/rebuild/status")
def rebuild_status() -> dict:
    """Get rebuild coordinator status."""
    coordinator = app_state.rebuild_coordinator
    if not coordinator:
        return {"status": "disabled", "reason": "no coordinator initialized"}
    return coordinator.get_status()


@app.get("/monitor/rebuild/jobs")
def rebuild_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """Get rebuild jobs."""
    coordinator = app_state.rebuild_coordinator
    if not coordinator:
        return {"jobs": []}

    filter_status = None
    if status:
        try:
            filter_status = RebuildStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    return {"jobs": coordinator.get_jobs(status=filter_status, limit=limit)}


@app.get("/monitor/rebuild/jobs/{job_id}")
def rebuild_job(job_id: str) -> dict:
    """Get a specific rebuild job."""
    coordinator = app_state.rebuild_coordinator
    if not coordinator:
        raise HTTPException(status_code=503, detail="Rebuild coordinator not initialized")

    job = coordinator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/monitor/rebuild/trigger")
def trigger_rebuild(
    project_id: str = Query(..., min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
    incremental: bool = Query(True, description="Use incremental rebuild"),
    changed_files: Optional[list[str]] = Query(None, description="List of changed files"),
) -> dict:
    """Manually trigger a rebuild for a project."""
    coordinator = app_state.rebuild_coordinator
    if not coordinator:
        raise HTTPException(status_code=503, detail="Rebuild coordinator not initialized")

    job = coordinator.queue_rebuild(
        project_id,
        trigger=RebuildTrigger.MANUAL,
        changed_files=changed_files,
        incremental=incremental,
    )
    return job.to_dict()


@app.post("/monitor/rebuild/jobs/{job_id}/cancel")
def cancel_rebuild(job_id: str) -> dict:
    """Cancel a pending rebuild job."""
    coordinator = app_state.rebuild_coordinator
    if not coordinator:
        raise HTTPException(status_code=503, detail="Rebuild coordinator not initialized")

    if coordinator.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}
    raise HTTPException(status_code=404, detail="Job not found or not cancellable")


# Project Status endpoint
@app.get("/projects/{project_id}/status")
def project_status(project_id: str) -> dict:
    """Get project indexing status."""
    coordinator = app_state.rebuild_coordinator
    project_path = _select_project(project_id)

    # Get current indexing status
    try:
        metadata, chunks = _load_project_index(project_path)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load project: {e}")

    sem = metadata.get("semantic_index", {})

    # Check if rebuild is in progress
    rebuilding = False
    rebuild_job = None
    if coordinator:
        jobs = coordinator.get_jobs()
        for job in jobs:
            if job["project_id"] == project_id and job["status"] in ("pending", "queued", "running"):
                rebuilding = True
                rebuild_job = job
                break

    return {
        "project_id": project_id,
        "indexed": True,
        "indexed_chunks": metadata.get("indexed_chunk_count", len(chunks)),
        "python_files": metadata.get("python_file_count", 0),
        "semantic_index": {
            "status": sem.get("status", "unavailable"),
            "model": sem.get("model_name"),
            "dimension": sem.get("dimension"),
            "window_count": sem.get("window_count"),
            "updated_at": sem.get("updated_at"),
            "error": sem.get("error"),
        },
        "retrieval_backend": metadata.get("retrieval_backend"),
        "rebuilding": rebuilding,
        "current_rebuild_job": rebuild_job,
        "last_updated": metadata.get("created_at"),
    }


# Cache Management endpoints
@app.get("/monitor/cache/stats")
def cache_stats() -> dict:
    """Get query cache statistics."""
    cache = app_state.query_cache
    if not cache:
        return {"status": "disabled", "reason": "no cache initialized"}
    return cache.get_stats()


@app.get("/monitor/cache/entries")
def cache_entries(
    project_id: Optional[str] = Query(None, min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Get cache entries for inspection."""
    cache = app_state.query_cache
    if not cache:
        return {"entries": []}
    return {"entries": cache.get_entries(project_id=project_id, limit=limit)}


@app.post("/monitor/cache/reset")
def reset_cache() -> dict:
    """Reset cache entries (destructive operation).

    Use with caution. Clears all or all project cache.
    """
    cache = app_state.query_cache
    if not cache:
        return {"status": "disabled", "reason": "no cache initialized"}

    count = cache.invalidate_all()
    return {"reset": True, "entries_cleared": count}


@app.post("/monitor/cache/invalidate")
def invalidate_cache(
    project_id: Optional[str] = Query(None, min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"),
) -> dict:
    """Invalidate cache entries."""
    cache = app_state.query_cache
    if not cache:
        return {"status": "disabled", "reason": "no cache initialized"}

    if project_id:
        count = cache.invalidate_project(project_id)
        return {"invalidated": count, "project_id": project_id}
    else:
        count = cache.invalidate_all()
        return {"invalidated": count, "all": True}


# Webhook endpoints
@app.get("/monitor/webhooks")
def list_webhooks() -> dict:
    """List registered webhooks."""
    dispatcher = app_state.webhook_dispatcher
    if not dispatcher:
        return {"webhooks": []}
    return {"webhooks": dispatcher.list_webhooks()}


@app.get("/monitor/webhooks/stats")
def webhook_stats() -> dict:
    """Get webhook dispatcher statistics."""
    dispatcher = app_state.webhook_dispatcher
    if not dispatcher:
        return {"status": "disabled"}
    return dispatcher.get_stats()


@app.get("/monitor/webhooks/history")
def webhook_history(
    event: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Get webhook delivery history."""
    dispatcher = app_state.webhook_dispatcher
    if not dispatcher:
        return {"history": []}

    event_type = None
    if event:
        try:
            event_type = WebhookEvent(event)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event}")

    return {"history": dispatcher.get_delivery_history(event=event_type, limit=limit)}


# System Metrics endpoints
@app.get("/monitor/metrics")
def system_metrics() -> dict:
    """Get system metrics."""
    collector = app_state.metrics_collector
    if not collector:
        return {"status": "disabled", "reason": "no metrics collector initialized"}

    return collector.get_system_health()


@app.get("/monitor/metrics/rebuilds")
def rebuild_metrics(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Get recent rebuild metrics."""
    collector = app_state.metrics_collector
    if not collector:
        return {"metrics": []}
    return {"metrics": collector.get_recent_rebuilds(limit)}


@app.get("/monitor/metrics/cache")
def cache_metrics(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Get recent cache operation metrics."""
    collector = app_state.metrics_collector
    if not collector:
        return {"metrics": []}
    return {"metrics": collector.get_recent_cache_ops(limit)}


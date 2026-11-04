"""Optional OpenAI-compatible generation with an offline evidence-only fallback."""

from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
    _has_dotenv = True
except ImportError:
    load_dotenv = None  # type: ignore[assignment]
    _has_dotenv = False

# Load root .env file for backend configuration
_env_loaded = False
if _has_dotenv and os.path.exists(".env") and not os.environ.get("SKIP_DOTENV"):
    load_dotenv()
    _env_loaded = True


_PYTHON_CITATION = re.compile(
    r"(?P<path>[^\s\[\]()]+\.py):(?P<first>\d+)(?:-(?P<last>\d+))?"
)


class LLMEngine:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        if not _env_loaded and _has_dotenv:
            load_dotenv()
        self.base_url = (base_url or os.getenv("TRACE_LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("TRACE_LLM_MODEL", "")
        self.api_key = api_key or os.getenv("TRACE_LLM_API_KEY", "")
        self.configured = bool(self.base_url and self.model)

    def is_configured(self) -> bool:
        """Check if LLM is properly configured without exposing credentials."""
        return self.configured

    def generate(self, prompt: str, source_chunks: list[dict] | None = None) -> str:
        if not self.base_url or not self.model:
            return self._evidence_fallback(source_chunks or [])
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}/chat/completions", data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            answer = str(data["choices"][0]["message"]["content"]).strip()
            if not self._citations_are_supported(answer, source_chunks or []):
                return self._evidence_fallback(source_chunks or [])
            return answer
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Configured LLM request failed: {exc}") from exc

    @staticmethod
    def _citations_are_supported(answer: str, chunks: list[dict]) -> bool:
        """Reject explicit Python citations outside retrieved source ranges."""
        citations = list(_PYTHON_CITATION.finditer(answer))
        if not citations:
            return True
        ranges: dict[str, list[tuple[int, int]]] = {}
        for chunk in chunks:
            path = str(chunk.get("file_path") or "")
            try:
                first = int(chunk["line_start"])
                last = int(chunk["line_end"])
            except (KeyError, TypeError, ValueError):
                continue
            ranges.setdefault(path, []).append((first, last))
        for citation in citations:
            path = citation.group("path")
            first = int(citation.group("first"))
            last = int(citation.group("last") or first)
            if first > last or not any(
                allowed_first <= first and last <= allowed_last
                for allowed_first, allowed_last in ranges.get(path, [])
            ):
                return False
        return True

    @staticmethod
    def _evidence_fallback(chunks: list[dict]) -> str:
        if not chunks:
            return "I could not find relevant indexed source code, so I cannot answer from this project."
        lines = ["No LLM is configured. Here are the closest indexed source excerpts:"]
        for item in chunks[:3]:
            lines.append(
                f"\n[{item.get('file_path', '?')}:{item.get('line_start', '?')}-{item.get('line_end', '?')}]\n"
                f"{str(item.get('code', '')).strip()}"
            )
        return "\n".join(lines)

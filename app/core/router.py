"""Small deterministic intent router; uncertain requests fall through to retrieval."""

from __future__ import annotations

import re


_FUNCTION = re.compile(r"(?:function|method)\s+[`'\"]?([\w.]+)", re.IGNORECASE)
_FILE = re.compile(r"(?:file|module)\s+[`'\"]?([\w./\\-]+\.py)", re.IGNORECASE)


def detect_intent(question: str) -> dict:
    """Detect the intent type from a user question.

    Returns:
        dict with 'type' (repo, inventory, file, function, usage, rag) and 'target' or None
    """
    text = question.strip()
    lowered = text.casefold()

    # Repository overview questions
    repo_keywords = [
        "explain the repo", "explain repository", "summarize this project", "what does this project",
        "what is this project about", "project architecture", "how the application works",
        "overview of the project", "tell me about this project", "what does this codebase do",
        "describe the architecture", "project structure", "application overview",
        "overview of the system", "system description"
    ]
    if any(phrase in lowered for phrase in repo_keywords):
        return {"type": "repo", "target": None}

    # File inventory/resource questions
    inventory_keywords = [
        "what are the different files", "list all files", "explain each file", "which files are important",
        "give me an overview of the project structure", "file list", "what files are in this project",
        "list all modules", "describe all files", "file inventory", "what files does this project have"
    ]
    if any(phrase in lowered for phrase in inventory_keywords):
        return {"type": "inventory", "target": None}

    # Function usage questions
    usage = _FUNCTION.search(text)
    if usage and any(word in lowered for word in ("used", "usage", "called", "calls", "reference", "references", "how is this used")):
        return {"type": "usage", "target": usage.group(1).rstrip("?.")}

    # Specific file questions
    file_match = _FILE.search(text)
    if file_match:
        return {"type": "file", "target": file_match.group(1).rstrip("?.")}

    # Function explanation questions
    function = _FUNCTION.search(text)
    if function and any(word in lowered for word in ("explain", "what does", "describe", "how does", "how works")):
        return {"type": "function", "target": function.group(1).rstrip("?.")}

    return {"type": "rag", "target": None}

"""Build a predictable, deterministic explanation from the repository summary."""


def build_repo_explanation(summary: str) -> str:
    if not summary.strip():
        return "The repository is empty or could not be summarized."
    return "Repository overview\n\n" + summary

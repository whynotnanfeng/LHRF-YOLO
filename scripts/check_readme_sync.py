"""Check that the two README files stay consistent with each other.

GitHub renders `.github/README.md` on the repository homepage and serves the
top-level `README.md` at `/blob`. They cover the same project, so the facts they
state about the paper, the dataset and the training configuration have to agree.
Keeping them in sync by hand has proven unreliable, so this is checked in CI.

Usage:
    python scripts/check_readme_sync.py
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Facts that must appear identically in both files.
SYNCED_FIELDS = {
    "paper doi": r"10\.3390/f16071095",
    "baidu mirror": r"pan\.baidu\.com/s/[A-Za-z0-9_-]+",
    "baidu code": r"\(code `m58i`\)",
    "google mirror": r"drive\.google\.com/file/d/[A-Za-z0-9_-]+",
    "quark mirror": r"pan\.quark\.cn/s/[A-Za-z0-9]+",
    "quark code": r"\(code `4aii`\)",
    "paper optimizer": r"optimizer=AdamW",
    "paper learning rate": r"lr0=0\.001",
    "paper batch size": r"batch=64",
}

# Absolute URLs, so the .github copy can link to the full docs.
EXPECTED_LINKS = ("README.md", "docs/DATASET.md")

# A Chinese character anywhere would break the "English only" rule.
CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\uff00-\uffef]")


def main() -> int:
    """Entry point. Returns a process exit code."""
    paths = {
        "README.md": REPO_ROOT / "README.md",
        ".github/README.md": REPO_ROOT / ".github" / "README.md",
    }

    failures = []
    texts = {}
    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        texts[name] = text
        for label, pattern in SYNCED_FIELDS.items():
            if not re.search(pattern, text):
                failures.append(f"{name}: {label} not found")
        if CJK.search(text):
            failures.append(f"{name}: contains CJK characters")
    if "README.md" not in texts or ".github/README.md" not in texts:
        print("\n".join(failures))
        return 1

    for link in EXPECTED_LINKS:
        if link not in texts[".github/README.md"]:
            failures.append(f".github/README.md: no link to {link}")

    if failures:
        print("README files are out of sync:")
        for failure in failures:
            print(f"  - {failure}")
        print("\n.github/README.md is what GitHub shows on the repository homepage.")
        return 1

    print("README files agree on: " + ", ".join(sorted(SYNCED_FIELDS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check the release-facing Markdown documentation contract."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_PATH = "results/README.md"
VALIDATION_ARTIFACTS_PATH = "results/controller-archive/README.md"
# These selected reports are intentionally committed and indexed from results/README.md.
CURATED_RESULT_DOCS = {
    "results/controller-archive/README.md",
    "results/headlight-v2-live/seed101-103-aggregate-20261005/all-metrics.md",
    "results/headlight-v2-live/seed101-103-aggregate-20261005/report.md",
}
REQUIRED = (
    "README.md",
    "docs/README.md",
    "docs/architecture.md",
    "docs/reproduction.md",
    "docs/experiments.md",
    "docs/evaluation.md",
    "docs/release.md",
    "docs/components/actions-handler.md",
    "docs/components/controlled-replica-propagation.md",
    "docs/components/ai-module.md",
    README_PATH,
    VALIDATION_ARTIFACTS_PATH,
)
REMOVED = (
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX_SOFTWAREX_ONE_SEED_INSTRUCTIONS.md",
    "HEADLIGHT_SUBMISSION_PLAN.md",
    "PUBLICATION_ROADMAP.md",
    "SOCC_ACTION_PLAN.md",
    "SOCC_PAPERS_TO_REVIEW.md",
    "REPRODUCTION.md",
    "SOFTWAREX_VALIDATION_RUNBOOK.md",
    "PERFORMANCE_VM_VALIDATION.md",
    "RELEASE_CHECKLIST.md",
)
PLACEHOLDERS = ("REPLACE-WITH-OWNER", "<repository-url>", "citation-tool")
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PUBLIC_DOCS = (
    Path("README.md"),
    Path("docs"),
    Path("actions-handler/README.md"),
    Path("controlled-replica-propagation/README.md"),
    Path("ai-module/README.md"),
    Path("ai-module/ai-module/README.md"),
    Path("experiments/README.md"),
    Path(README_PATH),
    Path(VALIDATION_ARTIFACTS_PATH),
)


def markdown_files() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(ROOT / name for name in tracked.stdout.splitlines() if (ROOT / name).is_file())


def tracked_files() -> set[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {name for name in tracked.stdout.splitlines() if (ROOT / name).is_file()}


def selected_public_docs() -> list[Path]:
    result: list[Path] = []
    for item in PUBLIC_DOCS:
        path = ROOT / item
        if path.is_dir():
            result.extend(sorted(path.rglob("*.md")))
        elif path.exists():
            result.append(path)
    return sorted(set(result))


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def check_required_docs(errors: list[str]) -> None:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            fail(f"missing canonical documentation: {relative}", errors)


def check_markdown_links(errors: list[str]) -> None:
    repository_files = tracked_files()
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for token in REMOVED + PLACEHOLDERS:
            if token in text:
                fail(f"{path.relative_to(ROOT)} contains forbidden token: {token}", errors)

        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().split()[0].split("#", 1)[0]
            if not target or re.match(r"(?:[a-z][a-z0-9+.-]*:|//)", target):
                continue
            if Path(target).is_absolute():
                fail(f"{path.relative_to(ROOT)} has non-portable absolute link: {raw_target}", errors)
                continue
            target = re.sub(r":\d+$", "", target)
            candidate = (path.parent / target).resolve()
            if not candidate.is_relative_to(ROOT):
                fail(f"{path.relative_to(ROOT)} has broken relative link: {raw_target}", errors)
                continue
            relative_candidate = candidate.relative_to(ROOT).as_posix()
            if relative_candidate not in repository_files and not any(
                tracked_path.startswith(relative_candidate.rstrip("/") + "/")
                for tracked_path in repository_files
            ):
                fail(f"{path.relative_to(ROOT)} has broken relative link: {raw_target}", errors)


def check_public_docs(errors: list[str]) -> None:
    for path in selected_public_docs():
        text = path.read_text(encoding="utf-8").lower()
        if "dry-run" not in text or "measured" not in text:
            fail(
                f"{path.relative_to(ROOT)} must distinguish dry-run validation from measured evidence",
                errors,
            )


def check_result_docs(errors: list[str]) -> None:
    result_markdown = [p for p in markdown_files() if Path("results") in p.relative_to(ROOT).parents]
    allowed_results = {
        ROOT / README_PATH,
        ROOT / VALIDATION_ARTIFACTS_PATH,
        *(ROOT / path for path in CURATED_RESULT_DOCS),
    }
    for path in result_markdown:
        if path not in allowed_results:
            fail(f"generated or historical result Markdown is tracked: {path.relative_to(ROOT)}", errors)


def main() -> int:
    errors: list[str] = []
    check_required_docs(errors)
    check_markdown_links(errors)
    check_public_docs(errors)
    check_result_docs(errors)

    if errors:
        print("Documentation check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Documentation check passed ({len(markdown_files())} Markdown files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

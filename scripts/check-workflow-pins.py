#!/usr/bin/env python3
"""Verify every SHA-pinned GitHub Action in .github/workflows/ resolves to a real commit.

A typo in a 40-char hex SHA is easy to make and only surfaces at workflow runtime
("Unable to resolve action ... unable to find version <sha>"). This script catches
the problem in CI before it reaches a release.

Reads GITHUB_TOKEN if set to use the authenticated API rate limit (5000 req/hr
vs 60 req/hr unauthenticated). In GitHub Actions the token is provided automatically.

Exits 0 if every pin is valid, 1 if any is missing.
"""

import os
import re
import sys
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path

PATTERN = re.compile(
    r"uses:\s*(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)"
    r"(?:/[A-Za-z0-9_./-]+)?@(?P<sha>[a-f0-9]{40})"
)


def check(owner: str, repo: str, sha: str, token: str | None) -> bool:
    """Return True if the SHA exists in the action repo, False if 404."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    # Hardcoded https://api.github.com host, no scheme injection possible from input.
    req = urllib.request.Request(  # noqa: S310
        url, headers={"Accept": "application/vnd.github+json"}
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return resp.status == HTTPStatus.OK
    except urllib.error.HTTPError as e:
        if e.code == HTTPStatus.NOT_FOUND:
            return False
        # Rate-limit or other API failure — surface it instead of pretending the pin is OK.
        print(f"  HTTP {e.code} for {owner}/{repo}@{sha}: {e.reason}", file=sys.stderr)
        raise


def main() -> int:
    workflows_dir = Path(".github/workflows")
    seen: set[tuple[str, str, str]] = set()
    for yml in sorted(workflows_dir.glob("*.yml")):
        for m in PATTERN.finditer(yml.read_text()):
            seen.add((m["owner"], m["repo"], m["sha"]))

    if not seen:
        print("No SHA-pinned actions found.")
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    print(f"Checking {len(seen)} SHA-pinned action reference(s)...")
    failed: list[tuple[str, str, str]] = []
    for owner, repo, sha in sorted(seen):
        ok = check(owner, repo, sha, token)
        status = "OK  " if ok else "FAIL"
        print(f"  {status}  {owner}/{repo}@{sha}")
        if not ok:
            failed.append((owner, repo, sha))

    if failed:
        print(f"\nERROR: {len(failed)} action pin(s) reference non-existent commits:")
        for owner, repo, sha in failed:
            print(f"  - {owner}/{repo}@{sha}")
        print(
            "\nFix: look up the correct SHA for the tag you want, e.g.\n"
            "  curl -s https://api.github.com/repos/<owner>/<repo>/git/refs/tags/v4 \\\n"
            "    | jq -r .object.sha"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

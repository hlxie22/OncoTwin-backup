#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/henryxie/OncoTwin-backup").resolve()
OUT = ROOT / "artifacts" / "checkpoint0b"

TOKENS = (
    "genie",
    "chord",
    "bpc",
    "brca",
    "aacr",
    "msk_chord",
    "msk-chord",
)

SKIP_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".cache",
}

# Avoid rediscovering thousands of known unrelated I-SPY2 files.
SKIP_PATH_FRAGMENTS = (
    "/data/ispy2_streaming/backups/",
    "/artifacts/checkpoint0/",
)


def human_gib(n: int) -> float:
    return round(n / 1024**3, 6)


def candidate_roots() -> list[Path]:
    # Do NOT recursively crawl the user's entire home or cluster scratch.
    # Large network filesystems can contain millions of directory entries.
    #
    # Search only the project's dedicated acquisition directory by default.
    # Additional known locations can be supplied explicitly through
    # ONCOTWIN_EXTRA_SEARCH_ROOTS.
    raw = [
        ROOT / "data" / "external_sources",
    ]

    extra = os.environ.get("ONCOTWIN_EXTRA_SEARCH_ROOTS", "")
    if extra:
        raw.extend(
            Path(x)
            for x in extra.split(os.pathsep)
            if x.strip()
        )

    roots: list[Path] = []

    for p in raw:
        try:
            p = p.expanduser().resolve()
        except Exception:
            continue

        if p.exists() and p.is_dir() and p not in roots:
            roots.append(p)

    return roots


def search_named_candidates(roots: list[Path]) -> list[dict[str, Any]]:
    """
    Filename/path search only. We deliberately do not scan contents of all
    large data files across scratch storage.
    """
    seen: set[str] = set()
    found: list[dict[str, Any]] = []

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_NAMES
            ]

            low_dir = str(dirpath).lower()

            if any(fragment in low_dir for fragment in SKIP_PATH_FRAGMENTS):
                dirnames[:] = []
                continue

            for name in filenames:
                path = Path(dirpath) / name
                low = str(path).lower()

                if not any(token in low for token in TOKENS):
                    continue

                try:
                    rp = str(path.resolve())
                    size = path.stat().st_size
                except OSError:
                    continue

                if rp in seen:
                    continue

                seen.add(rp)

                found.append({
                    "path": rp,
                    "size_bytes": size,
                    "size_gib": human_gib(size),
                })

    return sorted(found, key=lambda x: (-x["size_bytes"], x["path"]))


def repository_reference_search() -> list[str]:
    """
    Search source/config/docs for prior references to these datasets.
    Exclude bulk data and generated artifacts.
    """
    cmd = [
        "grep",
        "-R",
        "-I",
        "-n",
        "-E",
        "GENIE|MSK[-_ ]?CHORD|BPC|BrCa",
        "src",
        "scripts",
        "configs",
        "docs",
        ".",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
    except Exception as exc:
        return [f"grep failed: {type(exc).__name__}: {exc}"]

    rows = []

    for line in proc.stdout.splitlines():
        low = line.lower()

        if (
            "/data/" in low
            or "/artifacts/" in low
            or "/.git/" in low
        ):
            continue

        rows.append(line)

        if len(rows) >= 300:
            break

    return rows


def synapse_status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "client_importable": False,
        "authenticated": False,
        "genie_20_access": False,
        "bpc_public_project_access": False,
        "bpc_known_file_access": False,
        "bpc_release_ancestor_chain": [],
    }

    try:
        import synapseclient
        result["client_importable"] = True
    except Exception as exc:
        result["error"] = f"import: {type(exc).__name__}: {exc}"
        return result

    try:
        syn = synapseclient.login(silent=True)
        result["authenticated"] = True
    except Exception as exc:
        result["auth_error"] = f"{type(exc).__name__}: {exc}"
        return result

    for key, syn_id in (
        ("genie_20_access", "syn76285058"),
        ("bpc_public_project_access", "syn27056172"),
        ("bpc_known_file_access", "syn71825210"),
    ):
        try:
            syn.get(syn_id, downloadFile=False)
            result[key] = True
        except Exception as exc:
            result[key + "_error"] = f"{type(exc).__name__}: {exc}"

    if result["bpc_known_file_access"]:
        try:
            entity = syn.get("syn71825210", downloadFile=False)
            parent_id = getattr(entity, "parentId", None)

            depth = 0

            while parent_id and depth < 10:
                parent = syn.get(parent_id, downloadFile=False)

                result["bpc_release_ancestor_chain"].append({
                    "id": getattr(parent, "id", None),
                    "name": getattr(parent, "name", None),
                    "concreteType": getattr(parent, "concreteType", None),
                })

                parent_id = getattr(parent, "parentId", None)
                depth += 1

        except Exception as exc:
            result["bpc_ancestor_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    return result


def disk_status() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)

    return {
        "total_gib": round(usage.total / 1024**3, 3),
        "used_gib": round(usage.used / 1024**3, 3),
        "free_gib": round(usage.free / 1024**3, 3),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    roots = candidate_roots()

    print("[CKPT0B] filesystem candidate search starting")
    for root in roots:
        print(f"[CKPT0B] search_root={root}")

    candidates = search_named_candidates(roots)

    print(f"[CKPT0B] named_candidates={len(candidates)}")

    refs = repository_reference_search()
    syn = synapse_status()

    report = {
        "search_roots": [str(x) for x in roots],
        "disk": disk_status(),
        "local_candidates": candidates,
        "repository_references": refs,
        "synapse": syn,
    }

    json_path = OUT / "source_readiness.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    md: list[str] = []

    md.append("# Checkpoint 0B Source Readiness")
    md.append("")
    md.append("## Local candidate files")
    md.append("")

    if candidates:
        for x in candidates[:100]:
            md.append(
                f"- `{x['path']}` "
                f"({x['size_gib']} GiB)"
            )
    else:
        md.append("- No local GENIE/CHORD/BPC-named files found.")

    md.append("")
    md.append("## Synapse")
    md.append("")
    md.append(
        f"- client importable: `{syn['client_importable']}`"
    )
    md.append(
        f"- authenticated: `{syn['authenticated']}`"
    )
    md.append(
        f"- GENIE 20 access: `{syn['genie_20_access']}`"
    )
    md.append(
        "- BPC public project access: "
        f"`{syn['bpc_public_project_access']}`"
    )
    md.append(
        "- BPC BrCa known-file access: "
        f"`{syn['bpc_known_file_access']}`"
    )

    if syn["bpc_release_ancestor_chain"]:
        md.append("")
        md.append("### BPC BrCa release ancestor chain")
        md.append("")

        for x in syn["bpc_release_ancestor_chain"]:
            md.append(
                f"- `{x['id']}` — `{x['name']}` "
                f"({x['concreteType']})"
            )

    if syn.get("auth_error"):
        md.append("")
        md.append(
            f"- authentication error: `{syn['auth_error']}`"
        )

    md.append("")
    md.append("## Repository references")
    md.append("")

    if refs:
        for line in refs[:100]:
            md.append(f"- `{line}`")
    else:
        md.append("- None.")

    md_path = OUT / "source_readiness.md"
    md_path.write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    for p in (json_path, md_path):
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"missing output: {p}")

    print("")
    print("========== CKPT0B READINESS ==========")
    print(f"local_candidates={len(candidates)}")
    print(f"synapse_authenticated={syn['authenticated']}")
    print(f"genie20_access={syn['genie_20_access']}")
    print(
        "bpc_project_access="
        f"{syn['bpc_public_project_access']}"
    )
    print(
        "bpc_brca_file_access="
        f"{syn['bpc_known_file_access']}"
    )
    print(
        "bpc_ancestor_levels="
        f"{len(syn['bpc_release_ancestor_chain'])}"
    )
    print(
        f"free_gib={report['disk']['free_gib']}"
    )

    if candidates:
        print("largest_candidates:")
        for x in candidates[:20]:
            print(
                f"  {x['size_bytes']}\t{x['path']}"
            )

    if syn["bpc_release_ancestor_chain"]:
        print("bpc_ancestor_chain:")
        for x in syn["bpc_release_ancestor_chain"]:
            print(
                f"  {x['id']}\t{x['name']}\t{x['concreteType']}"
            )

    print(
        "report=artifacts/checkpoint0b/source_readiness.md"
    )
    print("========== CKPT0B READINESS END ==========")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

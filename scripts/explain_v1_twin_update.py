#!/usr/bin/env python3
"""Build a structured V1 explanation from posterior/scenario runtime artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.twin_runtime.explanations import (
    build_twin_update_explanation,
    render_markdown_explanation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build clinician/patient explanations from V1 twin runtime artifacts."
    )
    parser.add_argument("--posterior-update", required=True, type=Path)
    parser.add_argument("--scenario-lab", type=Path)
    parser.add_argument("--prior-context", type=Path)
    parser.add_argument("--audience", choices=("clinician", "patient"), default="clinician")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()

    explanation = build_twin_update_explanation(
        posterior_update=_read_json_object(args.posterior_update),
        scenario_lab=_read_json_object(args.scenario_lab) if args.scenario_lab else None,
        prior_context=_read_json_object(args.prior_context) if args.prior_context else None,
        audience=args.audience,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "markdown":
        args.output.write_text(render_markdown_explanation(explanation), encoding="utf-8")
    else:
        args.output.write_text(
            json.dumps(explanation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {args.audience} explanation to {args.output}")
    return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


if __name__ == "__main__":
    raise SystemExit(main())

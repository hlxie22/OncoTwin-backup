from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v2_06_external_protocol import (
    atomic_json,
    canonical_patient_token,
    load_json,
    overlap_report,
    require,
    validate_outcome_blind_manifest,
)


def read_csv_columns(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def read_ids(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and "patient_id" in reader.fieldnames,
                "identity CSV must contain patient_id")
        return [str(row["patient_id"]).strip() for row in reader if str(row["patient_id"]).strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Outcome-blind V2-06 new-cohort Stage-A preflight")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--upstream-token-file", help="Optional newline-delimited SHA256 identity tokens in the same authorized namespace")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    base = manifest_path.parent
    predictor_file = (base / manifest["predictor_inventory_file"]).resolve()
    identity_file = (base / manifest["patient_identity_file"]).resolve()

    cols = read_csv_columns(predictor_file)
    report = validate_outcome_blind_manifest(manifest, cols)

    ids = read_ids(identity_file)
    namespace = manifest["identity_namespace"]
    new_tokens = {canonical_patient_token(namespace, x) for x in ids}

    upstream_tokens: set[str] = set()
    if args.upstream_token_file:
        upstream_tokens = {x.strip() for x in Path(args.upstream_token_file).read_text().splitlines() if x.strip()}
    overlap = overlap_report(new_tokens, upstream_tokens)
    require(overlap["status"] == "PASS", f"Independent-cohort overlap check failed: {overlap['overlap_count']} overlapping patient tokens")

    report.update({
        "patient_count": len(new_tokens),
        "overlap": overlap,
        "outcomes_opened": False,
        "eligible_for_stage_b": bool(args.upstream_token_file),
        "stage_b_note": "Stage B additionally requires the V2-07 final deploy model bundle. No outcomes may be opened before prediction freeze."
    })
    atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

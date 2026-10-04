#!/usr/bin/env python3
"""Create a reference-only, research-only Layer 8 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


BUNDLE_VERSION = (
    "oncotwin_layer8_internal_evidence_bundle_v0_1"
)

EXPECTED_FILES = {
    "scripts/evals/run_layer8_ablation_uncertainty_screen_v0_3.py": {
        "sha256": "ef2aeabe368a79dbfe1022021f3e19754de9bae9635e4aab1be098fe697a8c15",
        "group": "candidate_selection",
        "role": "Parsimonious candidate and uncertainty screen implementation.",
    },
    "artifacts/prior_builder/layer8/layer8_ablation_uncertainty_screen_v0_3.json": {
        "sha256": "1331d9b5b30e8950690bfaa270bd63efc9c3d4635a91b256224471415a56bb26",
        "group": "candidate_selection",
        "role": "Candidate-selection and ablation results.",
    },
    "artifacts/prior_builder/layer8/layer8_ablation_uncertainty_screen_v0_3.md": {
        "sha256": "2d337de5456a08cbd97c3ea190ebc9e3dc7b4cda86d8730cb9036db2f472359c",
        "group": "candidate_selection",
        "role": "Human-readable candidate-selection report.",
    },
    "scripts/evals/run_layer8_strict_nested_internal.py": {
        "sha256": "159353a3792db9d565010a8a39024dff2ebab451151301efae9ba196e2268107",
        "group": "strict_internal_evaluation",
        "role": "Strict full-stack nested internal evaluation implementation.",
    },
    "artifacts/prior_builder/layer8/layer8_strict_nested_internal_v0_1_r5.json": {
        "sha256": "46b6417fd849f7c88bcacc7349f8b4fe8f302b6af85ce5dc6a62d79bdff7a5ba",
        "group": "strict_internal_evaluation",
        "role": "Strict nested internal result.",
    },
    "artifacts/prior_builder/layer8/layer8_strict_nested_internal_v0_1_r5.md": {
        "sha256": "09ec0ee1b0a86c5d6fe3f0cc87fc8c0130d1429791c84f06b18e81398872ce7b",
        "group": "strict_internal_evaluation",
        "role": "Human-readable strict nested report.",
    },
    "artifacts/prior_builder/layer8/layer8_strict_nested_internal_v0_1_r5.stdout.txt": {
        "sha256": "a29bd703d2bcc913ec4c7c5cb3340d8863f60bc85dbc2f6c44b03945307b14a5",
        "group": "strict_internal_evaluation",
        "role": "Validated strict-evaluation execution log.",
    },
    "data/processed/v1_prior_stack/ispy2_layer8_strict_nested_internal_repeat_predictions_v0_1_r5.jsonl": {
        "sha256": "7fc094976d9a7aa3754b5659bdf10344d663806d7251213f62cbf7d5928e6eca",
        "group": "strict_internal_evaluation",
        "role": "Case-by-repeat strict nested predictions.",
    },
    "data/processed/v1_prior_stack/ispy2_layer8_strict_nested_internal_case_summary_v0_1_r5.jsonl": {
        "sha256": "63eda3de8dc226748ad7ba9d0cbade7e62fbd1798b57600376a0ab8187eb501f",
        "group": "strict_internal_evaluation",
        "role": "Case-level strict nested summary.",
    },
    "scripts/evals/audit_layer8_executable_holdout_readiness.py": {
        "sha256": "541f8e0e6fdcb5207f29c89f8d918b971eaa4dac8c4dc79bb6881bccadabab23",
        "group": "validation_feasibility",
        "role": "Target-independence and executable-holdout audit.",
    },
    "artifacts/prior_builder/layer8/layer8_executable_holdout_readiness_v0_1.json": {
        "sha256": "5e8a497ea7e340c2148d6dfdd46b9f4e880bf5c447d1e651c0c9890dbd5a519e",
        "group": "validation_feasibility",
        "role": "Target-independence and initial holdout-readiness result.",
    },
    "artifacts/prior_builder/layer8/layer8_executable_holdout_readiness_v0_1.md": {
        "sha256": "fe9eb4916a06e9aa2f7609332419ac5953afd36465ff289369a2cdbb929591c0",
        "group": "validation_feasibility",
        "role": "Human-readable readiness report.",
    },
    "scripts/evals/audit_layer8_holdout_context_bridge_fast.py": {
        "sha256": "dad0f8b83f114ad56da8f9377e1437f6f724f87f7eb3566904e239e6d9bad34e",
        "group": "validation_feasibility",
        "role": "Bounded context-bridge audit implementation.",
    },
    "artifacts/prior_builder/layer8/layer8_holdout_context_bridge_fast_v0_1.json": {
        "sha256": "f58513c2680fb055e75070004e35afe119730ddc1bdedaf2c454dc56deb88a86",
        "group": "validation_feasibility",
        "role": "Bounded context-bridge result.",
    },
    "artifacts/prior_builder/layer8/layer8_holdout_context_bridge_fast_v0_1.md": {
        "sha256": "feae15d0bcc62ceb9ea5a0b25408d97d8d1c3ef4ce8ac015fda8b315864bc667",
        "group": "validation_feasibility",
        "role": "Human-readable context-bridge report.",
    },
    "scripts/evals/audit_layer8_contract_schema.py": {
        "sha256": "e01e7d6a497563b4feb3c42ba2014f887b6f4543b9733a45dedcac97469fb31a",
        "group": "population_accounting",
        "role": "Exact parameter-contract and subtype accounting implementation.",
    },
    "artifacts/prior_builder/layer8/layer8_contract_schema_audit_v0_1.json": {
        "sha256": "e403901953f2e7a844859b964f2d740a94d582049a9bce6a26666742d672b4bc",
        "group": "population_accounting",
        "role": "Exact same-source subtype and contract audit.",
    },
    "artifacts/prior_builder/layer8/layer8_contract_schema_audit_v0_1.md": {
        "sha256": "9171c2d5c24ee8f88eeb3d1e7553c58aa33de3f39d65213110339f91edcdc53b",
        "group": "population_accounting",
        "role": "Human-readable contract-schema report.",
    },
    "scripts/evals/finalize_layer8_internal_evidence_closure.py": {
        "sha256": "3d7ab1efd2312cac48b74a328646c2c036a4d1e74abc61372defbd3c8194cf8e",
        "group": "closure",
        "role": "Internal-evidence closure implementation.",
    },
    "artifacts/prior_builder/layer8/layer8_internal_evidence_closure_v0_1.json": {
        "sha256": "90573848e9e12389ee0bd94a3a6fa0714325b34c9d861d3a059bd28aa564f53b",
        "group": "closure",
        "role": "Authoritative Layer 8 internal-evidence closure.",
    },
    "artifacts/prior_builder/layer8/layer8_internal_evidence_closure_v0_1.md": {
        "sha256": "da480bf9dbbf8aac9b7d7bce9ac331565389d8847511051a6a1262dcb4006252",
        "group": "closure",
        "role": "Human-readable internal-evidence closure.",
    },
    "artifacts/prior_builder/layer8/layer8_external_validation_requirements_v0_1.json": {
        "sha256": "1296bfbe80ac6677398f7b84d84982fbd8378c21701f95fde14ead83dd61e1e1",
        "group": "external_validation",
        "role": "Prespecified external TNBC validation requirements.",
    },
    "artifacts/prior_builder/layer8/layer8_external_validation_requirements_v0_1.md": {
        "sha256": "3c9546c8cdfa4e1e88f4a51c2bc05dc5349ec83d2c83d49f86161e168dfc38f1",
        "group": "external_validation",
        "role": "Human-readable external-validation requirements.",
    },
    "data/processed/v1_prior_stack/ispy2_layer8_same_source_ineligibility_v0_1.jsonl": {
        "sha256": "73f33f6626ae4d73668af5d908fdf2e38f46250cda14170797ee859e14aa8c90",
        "group": "population_accounting",
        "role": "Target-free accounting of the 435 non-TNBC same-source cases.",
    },
    "artifacts/prior_builder/layer8/layer8_internal_evidence_closure_v0_1.stdout.txt": {
        "sha256": "44bb37e58223d94af3710cb1f022dfb4cf2143ff7033d9f9b31c1613e066b2eb",
        "group": "closure",
        "role": "Validated closure execution log.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--readme",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lock-marker",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, Mapping):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return dict(value)


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def main() -> None:
    args = parse_args()

    entries = []

    for raw_path, metadata in sorted(
        EXPECTED_FILES.items()
    ):
        path = Path(raw_path)

        if not path.is_file():
            raise RuntimeError(
                f"Required evidence file is missing: {path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Required evidence file is empty: {path}"
            )

        actual_hash = sha256_file(path)

        if actual_hash != metadata["sha256"]:
            raise RuntimeError(
                f"Evidence hash mismatch: {path} "
                f"expected={metadata['sha256']} "
                f"actual={actual_hash}"
            )

        entries.append(
            {
                "path": raw_path,
                "sha256": actual_hash,
                "bytes": path.stat().st_size,
                "group": metadata["group"],
                "role": metadata["role"],
            }
        )

    closure_path = Path(
        "artifacts/prior_builder/layer8/"
        "layer8_internal_evidence_closure_v0_1.json"
    )
    requirements_path = Path(
        "artifacts/prior_builder/layer8/"
        "layer8_external_validation_requirements_v0_1.json"
    )

    closure = load_json(closure_path)
    requirements = load_json(
        requirements_path
    )

    decision = closure[
        "scientific_decision"
    ]

    required_decision = {
        "layer6_remains_primary_predictor": True,
        "layer8_promoted": False,
        "layer8_deployment_allowed": False,
        "layer8_model_lock_allowed": False,
        "same_source_validation_allowed": False,
        "external_tnbc_validation_required": True,
        "non_tnbc_generalization_claim_allowed": False,
    }

    for key, expected in (
        required_decision.items()
    ):
        actual = decision.get(key)

        if actual != expected:
            raise RuntimeError(
                f"Closure decision mismatch: "
                f"{key} expected={expected} actual={actual}"
            )

    accounting = closure[
        "same_source_population_accounting"
    ]

    if accounting[
        "outside_tnbc_case_count"
    ] != 0:
        raise RuntimeError(
            "Closure unexpectedly reports an unused TNBC case"
        )

    if accounting[
        "same_source_tnbc_holdout_available"
    ]:
        raise RuntimeError(
            "Closure unexpectedly permits same-source validation"
        )

    if requirements[
        "claim_boundary"
    ]["same_source_validation_available"]:
        raise RuntimeError(
            "External requirements permit same-source validation"
        )

    if not requirements[
        "claim_boundary"
    ]["external_validation_required"]:
        raise RuntimeError(
            "External-validation requirement is absent"
        )

    groups: dict[str, list[str]] = {}

    for entry in entries:
        groups.setdefault(
            entry["group"],
            [],
        ).append(entry["path"])

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "status": (
            "research_only_internal_evidence_bundle"
        ),
        "candidate": {
            "name": "ridge_t2_prior_only",
            "new_evidence": "T2",
            "layer6_information_cutoff": "T1",
            "internal_target": "T3",
            "status": (
                "frozen_research_candidate"
            ),
        },
        "authoritative_closure": {
            "path": str(closure_path),
            "sha256": sha256_file(
                closure_path
            ),
        },
        "scientific_status": {
            "layer6_remains_primary_predictor": True,
            "layer8_promoted": False,
            "layer8_deployment_allowed": False,
            "layer8_model_lock_allowed": False,
            "same_source_tnbc_holdout_available": False,
            "external_tnbc_validation_required": True,
            "independent_confirmation_available": False,
            "non_tnbc_generalization_claim_allowed": False,
        },
        "internal_evidence": {
            "development_tnbc_cases": 270,
            "strict_nested_landmark_cases": 220,
            "strict_nested_repeats": 5,
            "no_update_log_rmse": (
                closure[
                    "strict_nested_internal_evidence"
                ]["metrics"]["no_update"]["log_rmse"]
            ),
            "candidate_log_rmse": (
                closure[
                    "strict_nested_internal_evidence"
                ]["metrics"][
                    "ridge_t2_prior_only"
                ]["log_rmse"]
            ),
            "candidate_vs_no_update_delta_log_rmse": (
                closure[
                    "strict_nested_internal_evidence"
                ]["paired_comparisons"][
                    "candidate_vs_no_update"
                ]["delta_log_rmse"]
            ),
            "coverage80": (
                closure[
                    "strict_nested_internal_evidence"
                ]["uncertainty"]["coverage80"]
            ),
            "coverage95": (
                closure[
                    "strict_nested_internal_evidence"
                ]["uncertainty"]["coverage95"]
            ),
        },
        "same_source_accounting": {
            "full_cases": 705,
            "development_tnbc_cases": 270,
            "outside_cases": 435,
            "outside_tnbc_cases": 0,
            "outside_subtypes": (
                accounting[
                    "outside_subtype_counts"
                ]
            ),
        },
        "evidence_groups": {
            group: sorted(paths)
            for group, paths
            in sorted(groups.items())
        },
        "files": entries,
        "prohibited_uses": [
            (
                "Do not treat this bundle as a "
                "deployable Layer 8 runtime."
            ),
            (
                "Do not claim external or independent "
                "validation from this bundle."
            ),
            (
                "Do not use the 435 non-TNBC cases as "
                "confirmatory validation for the TNBC candidate."
            ),
            (
                "Do not perform additional candidate selection "
                "or hyperparameter tuning using the same "
                "220 Layer 8 landmark cases."
            ),
        ],
        "next_allowed_modeling_step": (
            "One prespecified evaluation on an independent "
            "external TNBC cohort satisfying the frozen "
            "external-validation requirements."
        ),
    }

    readme = "\n".join(
        [
            "# Layer 8 internal evidence bundle",
            "",
            "## Status",
            "",
            (
                "**Research only. Not deployable. "
                "Not independently validated.**"
            ),
            "",
            (
                "This directory is a reference-only closure "
                "bundle. It does not contain a Layer 8 runtime "
                "and does not promote Layer 8 over the frozen "
                "Layer 6 predictor."
            ),
            "",
            "## Internal finding",
            "",
            (
                "The selected `ridge_t2_prior_only` updater "
                "survived strict full-stack nested internal "
                "evaluation on 220 TNBC landmark cases."
            ),
            "",
            (
                "Internal log RMSE improved from "
                f"`{manifest['internal_evidence']['no_update_log_rmse']:.6f}` "
                "for no update to "
                f"`{manifest['internal_evidence']['candidate_log_rmse']:.6f}` "
                "for the candidate."
            ),
            "",
            "## Validation boundary",
            "",
            (
                "The 270-case development cohort exhausts the "
                "TNBC patients in the 705-case same-source "
                "dataset. The remaining 435 patients are "
                "non-TNBC and are outside the frozen parameter "
                "contract."
            ),
            "",
            (
                "Layer 6 therefore remains the primary "
                "deployable predictor. Independent external "
                "TNBC validation is required before Layer 8 "
                "can be promoted."
            ),
            "",
            "## Integrity",
            "",
            (
                "`manifest.json` records the scientific state "
                "and evidence chain. `checksums.json` records "
                "the exact hashes and sizes of every referenced "
                "artifact and of this bundle's generated files."
            ),
        ]
    ) + "\n"

    lock_text = "\n".join(
        [
            "RESEARCH_ONLY",
            "LAYER6_REMAINS_PRIMARY=true",
            "LAYER8_DEPLOYMENT_ALLOWED=false",
            "LAYER8_MODEL_LOCK_ALLOWED=false",
            "SAME_SOURCE_TNBC_VALIDATION_AVAILABLE=false",
            "EXTERNAL_TNBC_VALIDATION_REQUIRED=true",
            "ADDITIONAL_DEVELOPMENT_TUNING_ALLOWED=false",
            "",
        ]
    )

    atomic_write_json(
        args.manifest,
        manifest,
    )
    atomic_write_text(
        args.readme,
        readme,
    )
    atomic_write_text(
        args.lock_marker,
        lock_text,
    )

    generated_files = {
        str(args.manifest): {
            "sha256": sha256_file(
                args.manifest
            ),
            "bytes": args.manifest.stat().st_size,
        },
        str(args.readme): {
            "sha256": sha256_file(
                args.readme
            ),
            "bytes": args.readme.stat().st_size,
        },
        str(args.lock_marker): {
            "sha256": sha256_file(
                args.lock_marker
            ),
            "bytes": args.lock_marker.stat().st_size,
        },
    }

    checksums = {
        "bundle_version": BUNDLE_VERSION,
        "status": "complete",
        "source_files": {
            entry["path"]: {
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
            }
            for entry in entries
        },
        "generated_bundle_files": (
            generated_files
        ),
    }

    atomic_write_json(
        args.checksums,
        checksums,
    )

    print(
        "LAYER8_BUNDLE_SOURCE_FILES "
        f"count={len(entries)}"
    )
    print(
        "LAYER8_BUNDLE_EVIDENCE_GROUPS "
        f"groups={sorted(groups)}"
    )
    print(
        "LAYER8_BUNDLE_SCIENTIFIC_STATUS "
        "layer6_primary=true "
        "layer8_promoted=false "
        "layer8_deployable=false "
        "same_source_validation=false "
        "external_tnbc_validation_required=true"
    )
    print(
        "LAYER8_BUNDLE_TUNING_GUARD "
        "additional_same_data_tuning_allowed=false"
    )
    print(
        f"LAYER8_BUNDLE_MANIFEST path={args.manifest}"
    )
    print(
        f"LAYER8_BUNDLE_CHECKSUMS path={args.checksums}"
    )
    print(
        f"LAYER8_BUNDLE_README path={args.readme}"
    )
    print(
        f"LAYER8_BUNDLE_LOCK_MARKER "
        f"path={args.lock_marker}"
    )


if __name__ == "__main__":
    main()

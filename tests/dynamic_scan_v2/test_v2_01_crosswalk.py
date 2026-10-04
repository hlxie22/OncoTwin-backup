from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "dynamic_scan_v2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v2_01_build_scan_representation import _legacy_text_surface_features


def test_legacy_combined_surface_can_cross_activate_modality_from_other_source():
    frame = pd.DataFrame(
        {
            "modalities_json": ['[]', '["CT"]', '["MRI"]'],
            "tumor_sites_json": ['["PET"]', '["LIVER"]', '["BONE SCAN"]'],
            "unrelated": ["PET", "MRI", "BONE SCAN"],
        }
    )
    combined, names, candidates = _legacy_text_surface_features(frame)
    modality_only, _, _ = _legacy_text_surface_features(
        frame, candidate_columns=["modalities_json"]
    )

    assert names[:4] == [
        "modality_ct", "modality_pet", "modality_mr", "modality_bone_scan"
    ]
    # Historical CKPT5 bug: its literal token "modality" does not match the
    # canonical plural field name "modalities_json". Only tumor_sites_json is
    # selected from this fixture; unrelated text is excluded as intended.
    assert candidates == ["tumor_sites_json"]

    # Row 0: legacy text sees PET in tumor_sites_json even though the actual
    # modality source is empty.
    assert combined[0, 1] == 1.0
    assert modality_only[0, 1] == 0.0

    # Row 1: actual CT modality is invisible to the historical selector.
    assert combined[1, 0] == 0.0
    assert modality_only[1, 0] == 1.0

    # Row 2: BONE SCAN in tumor-sites can spuriously activate legacy modality,
    # while the actual MRI modality is missed by that selector.
    assert combined[2, 3] == 1.0
    assert modality_only[2, 3] == 0.0
    assert combined[2, 2] == 0.0
    assert modality_only[2, 2] == 1.0


def test_source_specific_legacy_reconstruction_keeps_site_and_modality_separate():
    frame = pd.DataFrame(
        {
            "modalities_json": ['["BONE SCAN"]', '["PET-CT"]'],
            "tumor_sites_json": ['[]', '["BRAIN"]'],
        }
    )
    modality_surface, _, _ = _legacy_text_surface_features(
        frame, candidate_columns=["modalities_json"]
    )
    site_surface, _, _ = _legacy_text_surface_features(
        frame, candidate_columns=["tumor_sites_json"]
    )

    # BONE SCAN is modality evidence only under the V2 provenance rule.
    assert modality_surface[0, 3] == 1.0
    assert site_surface[0, 4] == 0.0

    # BRAIN is site-positive evidence and cannot create modality evidence.
    assert site_surface[1, 7] == 1.0  # site_brain
    assert not site_surface[1, :4].any()

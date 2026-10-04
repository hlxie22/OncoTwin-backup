from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import os

import numpy as np

from v2_07_inference import OncoTwinV2Predictor, batch_from_npz, jsonable_summary


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verify-repo", default=None)
    args = p.parse_args()

    predictor = OncoTwinV2Predictor(args.checkpoint, device=args.device)
    if args.verify_repo:
        predictor.verify_upstream_files(args.verify_repo)
    batch = batch_from_npz(args.input)
    result = predictor.predict(batch)
    curves = result["selected_curves"]
    atomic_npz(
        Path(args.output),
        supported=result["supported"].astype(np.uint8),
        pre_logits=result["pre_logits"],
        post_logits=result["post_logits"],
        selected_logits=result["selected_logits"],
        pre_conditional_probabilities=result["pre_conditional_probabilities"],
        post_conditional_probabilities=result["post_conditional_probabilities"],
        selected_conditional_probabilities=result["selected_conditional_probabilities"],
        event_free_survival=curves["event_free_survival"],
        progression_cif=curves["progression_cif"],
        death_cif=curves["death_cif"],
        switch_cif=curves["switch_cif"],
        progression_or_death_cif=curves["progression_or_death_cif"],
        pfs_survival=curves["pfs_survival"],
        horizon_progression_or_death_cif=result["selected_horizon_progression_or_death_cif"],
        horizon_pfs_survival=result["selected_horizon_pfs_survival"],
        horizon_switch_cif=result["selected_horizon_switch_cif"],
        horizon_event_free_survival=result["selected_horizon_event_free_survival"],
    )
    print(json.dumps(jsonable_summary(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

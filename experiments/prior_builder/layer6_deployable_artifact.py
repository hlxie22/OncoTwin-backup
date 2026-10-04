"""Safe serialization helpers for the deployable Layer 6 neural ensemble."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.prior_builder.layer6_neural_inference import (
    FittedLayer6NeuralModel,
    LAYER6_NEURAL_VERSION,
    Layer6NeuralConfig,
    Layer6ProbabilisticMLP,
)


LAYER6_DEPLOYABLE_ARTIFACT_VERSION = (
    "oncotwin_layer6_deployable_ensemble_v1"
)
LAYER6_MEMBER_FORMAT_VERSION = 1


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write deterministic JSON without permitting NaN or partial output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"

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

    temporary.replace(path)


def save_fitted_layer6_member(
    fitted: FittedLayer6NeuralModel,
    path: Path,
    *,
    member_index: int,
    feature_names: Sequence[str],
) -> dict[str, object]:
    """Save a fitted member as tensors plus primitive metadata only."""

    if member_index < 0:
        raise ValueError("member_index must be nonnegative")

    fitted.config.validate()

    names = tuple(str(name) for name in feature_names)
    feature_mean = np.asarray(fitted.feature_mean, dtype=float).reshape(-1)
    feature_sd = np.asarray(fitted.feature_sd, dtype=float).reshape(-1)

    if not names:
        raise ValueError("feature_names must not be empty")
    if len(names) != len(feature_mean):
        raise ValueError(
            "feature_names length must match feature normalization length"
        )
    if feature_sd.shape != feature_mean.shape:
        raise ValueError("feature_mean and feature_sd must have equal shape")
    if not np.isfinite(feature_mean).all():
        raise ValueError("feature_mean contains non-finite values")
    if not np.isfinite(feature_sd).all() or np.any(feature_sd <= 0.0):
        raise ValueError("feature_sd must contain finite positive values")
    if not np.isfinite(float(fitted.target_mean)):
        raise ValueError("target_mean must be finite")
    if not np.isfinite(float(fitted.target_sd)) or float(fitted.target_sd) <= 0.0:
        raise ValueError("target_sd must be finite and positive")

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in fitted.model.state_dict().items()
    }

    payload: dict[str, object] = {
        "format_version": LAYER6_MEMBER_FORMAT_VERSION,
        "artifact_version": LAYER6_DEPLOYABLE_ARTIFACT_VERSION,
        "neural_model_version": LAYER6_NEURAL_VERSION,
        "member_index": int(member_index),
        "seed": int(fitted.seed),
        "feature_names": list(names),
        "input_dim": len(names),
        "config": asdict(fitted.config),
        "feature_mean": feature_mean.tolist(),
        "feature_sd": feature_sd.tolist(),
        "target_mean": float(fitted.target_mean),
        "target_sd": float(fitted.target_sd),
        "best_epoch": int(fitted.best_epoch),
        "best_validation_loss": (
            None
            if fitted.best_validation_loss is None
            else float(fitted.best_validation_loss)
        ),
        "state_dict": state_dict,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "member_index": int(member_index),
        "file": path.name,
        "sha256": sha256_file(path),
        "seed": int(fitted.seed),
        "feature_mean": feature_mean.tolist(),
        "feature_sd": feature_sd.tolist(),
        "target_mean": float(fitted.target_mean),
        "target_sd": float(fitted.target_sd),
        "best_epoch": int(fitted.best_epoch),
        "best_validation_loss": (
            None
            if fitted.best_validation_loss is None
            else float(fitted.best_validation_loss)
        ),
    }


def load_fitted_layer6_member(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_feature_names: Sequence[str] | None = None,
) -> FittedLayer6NeuralModel:
    """Load a state-dict-only member with strict hash and schema checks."""

    if expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Layer 6 member hash mismatch for {path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "This Layer 6 artifact loader requires a PyTorch version "
            "that supports torch.load(..., weights_only=True)."
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("Layer 6 member payload must be a mapping")

    required = {
        "format_version",
        "artifact_version",
        "neural_model_version",
        "member_index",
        "seed",
        "feature_names",
        "input_dim",
        "config",
        "feature_mean",
        "feature_sd",
        "target_mean",
        "target_sd",
        "best_epoch",
        "best_validation_loss",
        "state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "Layer 6 member payload is missing fields: " + ", ".join(missing)
        )

    if int(payload["format_version"]) != LAYER6_MEMBER_FORMAT_VERSION:
        raise ValueError("unsupported Layer 6 member format_version")
    if payload["artifact_version"] != LAYER6_DEPLOYABLE_ARTIFACT_VERSION:
        raise ValueError("unexpected Layer 6 artifact version")
    if payload["neural_model_version"] != LAYER6_NEURAL_VERSION:
        raise ValueError("unexpected Layer 6 neural model version")

    feature_names = tuple(str(name) for name in payload["feature_names"])
    if expected_feature_names is not None:
        expected = tuple(str(name) for name in expected_feature_names)
        if feature_names != expected:
            raise ValueError(
                "Layer 6 feature order mismatch: "
                f"expected {expected}, got {feature_names}"
            )

    input_dim = int(payload["input_dim"])
    if input_dim < 1 or input_dim != len(feature_names):
        raise ValueError("invalid Layer 6 member input_dim")

    config_payload = payload["config"]
    if not isinstance(config_payload, Mapping):
        raise ValueError("Layer 6 member config must be a mapping")
    config = Layer6NeuralConfig(
        **{str(key): value for key, value in config_payload.items()}
    )
    config.validate()

    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_sd = np.asarray(payload["feature_sd"], dtype=np.float32)
    if feature_mean.shape != (input_dim,) or feature_sd.shape != (input_dim,):
        raise ValueError("invalid Layer 6 normalization shape")
    if not np.isfinite(feature_mean).all():
        raise ValueError("feature_mean contains non-finite values")
    if not np.isfinite(feature_sd).all() or np.any(feature_sd <= 0.0):
        raise ValueError("feature_sd must contain finite positive values")

    target_mean = float(payload["target_mean"])
    target_sd = float(payload["target_sd"])
    if not np.isfinite(target_mean):
        raise ValueError("target_mean must be finite")
    if not np.isfinite(target_sd) or target_sd <= 0.0:
        raise ValueError("target_sd must be finite and positive")

    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("Layer 6 state_dict must be a mapping")

    model = Layer6ProbabilisticMLP(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()

    best_validation_loss_raw = payload["best_validation_loss"]
    best_validation_loss = (
        None
        if best_validation_loss_raw is None
        else float(best_validation_loss_raw)
    )

    return FittedLayer6NeuralModel(
        model=model,
        feature_mean=feature_mean,
        feature_sd=feature_sd,
        target_mean=target_mean,
        target_sd=target_sd,
        best_epoch=int(payload["best_epoch"]),
        best_validation_loss=best_validation_loss,
        config=config,
        seed=int(payload["seed"]),
    )

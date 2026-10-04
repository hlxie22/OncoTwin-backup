from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class AccessDeniedError(RuntimeError):
    pass


@dataclass(frozen=True)
class AccessPolicy:
    repo: Path
    blocked_path_tokens: tuple[str, ...]
    blocked_basenames: tuple[str, ...]
    blocked_relative_roots: tuple[str, ...]
    allowed_schema_positive_controls: tuple[str, ...]

    @classmethod
    def from_json(cls, repo: Path, path: Path) -> "AccessPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            repo=Path(repo).resolve(),
            blocked_path_tokens=tuple(x.lower() for x in payload["blocked_path_tokens"]),
            blocked_basenames=tuple(x.lower() for x in payload["blocked_basenames"]),
            blocked_relative_roots=tuple(x.lower().rstrip("/") for x in payload["blocked_relative_roots"]),
            allowed_schema_positive_controls=tuple(
                x.lower().lstrip("./") for x in payload.get("allowed_schema_positive_controls", [])
            ),
        )

    def relative_display(self, path: Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            return resolved.relative_to(self.repo).as_posix()
        except ValueError:
            return resolved.as_posix()

    def assert_allowed(self, path: Path, purpose: str = "read") -> Path:
        resolved = Path(path).expanduser().resolve()
        display = self.relative_display(resolved).lower().lstrip("./")
        basename = resolved.name.lower()

        if display in self.allowed_schema_positive_controls:
            return resolved

        for root in self.blocked_relative_roots:
            if display == root or display.startswith(root + "/"):
                raise AccessDeniedError(
                    f"V2 access policy blocked {purpose}: {display} (blocked root {root})"
                )

        if basename in self.blocked_basenames:
            raise AccessDeniedError(
                f"V2 access policy blocked {purpose}: {display} (blocked outcome filename)"
            )

        for token in self.blocked_path_tokens:
            if token and token in display:
                raise AccessDeniedError(
                    f"V2 access policy blocked {purpose}: {display} (blocked token {token})"
                )

        return resolved

    def read_json(self, path: Path) -> dict[str, Any]:
        allowed = self.assert_allowed(path, "JSON read")
        return json.loads(allowed.read_text(encoding="utf-8"))

    def read_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        allowed = self.assert_allowed(path, "Parquet read")
        return pd.read_parquet(allowed, **kwargs)

    def np_load(self, path: Path, **kwargs):
        allowed = self.assert_allowed(path, "NumPy read")
        return np.load(allowed, **kwargs)

    def torch_load(self, path: Path, *, torch_module, **kwargs):
        allowed = self.assert_allowed(path, "Torch checkpoint read")
        return torch_module.load(allowed, **kwargs)

"""Safety language checks for simulator outputs."""

from __future__ import annotations

import json
from typing import Any


FORBIDDEN_PHRASES = [
    "will cure",
    "guaranteed",
    "best treatment",
    "you should take",
    "clinical pcr prediction",
    "cancer is gone",
    "this treatment will work",
    "this treatment will cure",
    "you should choose this treatment",
    "guaranteed response",
]


def find_unsafe_language(payload: Any) -> list[str]:
    text = json.dumps(payload, sort_keys=True).lower()
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in text]


def assert_no_unsafe_language(payload: Any) -> None:
    matches = find_unsafe_language(payload)
    if matches:
        raise AssertionError(f"unsafe simulator language found: {matches}")

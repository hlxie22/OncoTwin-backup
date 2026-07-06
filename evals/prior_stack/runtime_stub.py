"""CLI helper for eval categories whose runtime is not implemented yet."""
from __future__ import annotations

import argparse
from typing import Sequence

from .common import EvalUnavailable, print_result, runtime_stub_result


def main_for(
    name: str,
    modules: Sequence[str],
    summary: str,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise instead of returning unavailable status.",
    )
    args = parser.parse_args(argv)

    try:
        result = runtime_stub_result(name, modules, summary, strict=args.strict)
    except EvalUnavailable as exc:
        result = exc.result()

    print_result(result)
    return 0 if result.available else 2

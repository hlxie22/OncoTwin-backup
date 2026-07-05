"""File and fixture utilities for simulator experiment runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "mechanistic_simulator" / "outputs"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_case_and_schedule(case_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    case = load_json(case_path)
    schedule_path = REPO_ROOT / case["treatment_schedule"]["path"]
    return case, load_json(schedule_path)


def output_days(total_duration_days: int | float, step_days: int = 7) -> list[int]:
    total = int(total_duration_days)
    days = list(range(0, total + 1, step_days))
    if days[-1] != total:
        days.append(total)
    return days


def write_svg_trajectory_plot(
    path: str | Path,
    times: list[float],
    median: list[float],
    interval_80: list[list[float]],
    title: str,
) -> None:
    """Write a small dependency-free SVG trajectory plot."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 820
    height = 460
    margin_left = 70
    margin_right = 30
    margin_top = 44
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_time = max(times) if times else 1.0
    max_volume = max([value for pair in interval_80 for value in pair] + median + [1.0])

    def sx(day: float) -> float:
        return margin_left + (float(day) / max_time) * plot_width if max_time else margin_left

    def sy(volume: float) -> float:
        return margin_top + plot_height - (float(volume) / max_volume) * plot_height

    upper = [(sx(day), sy(pair[1])) for day, pair in zip(times, interval_80)]
    lower = [(sx(day), sy(pair[0])) for day, pair in reversed(list(zip(times, interval_80)))]
    band_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
    median_points = " ".join(f"{sx(day):.1f},{sy(value):.1f}" for day, value in zip(times, median))

    escaped_title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{margin_left}" y="28" font-family="Arial, sans-serif" font-size="18" fill="#17202a">{escaped_title}</text>
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#5d6d7e"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#5d6d7e"/>
  <text x="{margin_left + plot_width / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#17202a">Day</text>
  <text x="16" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 16,{margin_top + plot_height / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#17202a">Tumor volume (mL)</text>
  <polygon points="{band_points}" fill="#9ecae1" opacity="0.45"/>
  <polyline points="{median_points}" fill="none" stroke="#0b4f6c" stroke-width="3"/>
  <text x="{width - margin_right}" y="{margin_top + 18}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#17202a">Band: model-estimated 80% interval</text>
  <text x="{width - margin_right}" y="{margin_top + 36}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#17202a">Line: median exploratory simulation</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")

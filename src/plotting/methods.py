"""Shared method definitions for the main XS/S/M/L plots and table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MAIN_SCALE_SIZE_SUFFIX = {
    "xs": "20b",
    "s": "20b",
    "m": "20b",
    "l": "20b",
    "xl": "20b",
}


@dataclass(frozen=True)
class MainMethodSpec:
    key: str
    label: str
    family: str
    window: int | None


MAIN_METHOD_SPECS = (
    MainMethodSpec("standard", "Standard", "full_attention", None),
    MainMethodSpec("two_x_memory", "2x Memory", "sps", 4096),
    MainMethodSpec("delayed_state", "Delayed State", "delayed_state", 64),
    MainMethodSpec("sps", "SPS", "sps", 64),
)

MAIN_METHOD_ORDER = {method.key: idx for idx, method in enumerate(MAIN_METHOD_SPECS)}


def main_size_suffix_for_scale(scale: str) -> str:
    try:
        return MAIN_SCALE_SIZE_SUFFIX[scale]
    except KeyError as exc:
        raise ValueError(f"Unsupported scale {scale!r}") from exc


def main_run_name(scale: str, method: MainMethodSpec) -> str:
    size_suffix = main_size_suffix_for_scale(scale)
    if method.family == "full_attention":
        return f"{scale}_full_attention_{size_suffix}"
    if method.window is None:
        raise ValueError(f"Windowed method {method.key!r} requires a window")
    return f"{scale}_{method.family}_w{int(method.window)}_{size_suffix}"


def main_method_for_family_window(family: str, window: int | None) -> MainMethodSpec | None:
    normalized_window = None if window is None else int(window)
    for method in MAIN_METHOD_SPECS:
        if method.family == family and method.window == normalized_window:
            return method
    return None


def main_method_specs_for_selection(
    *,
    families: Iterable[str] | None = None,
    windows: Iterable[int] | None = None,
    include_standard: bool = True,
) -> tuple[MainMethodSpec, ...]:
    family_filter = None if families is None else {str(family) for family in families}
    window_filter = None if windows is None else {int(window) for window in windows}
    selected: list[MainMethodSpec] = []
    for method in MAIN_METHOD_SPECS:
        if method.family == "full_attention":
            if include_standard:
                selected.append(method)
            continue
        if family_filter is not None and method.family not in family_filter:
            continue
        if window_filter is not None and int(method.window) not in window_filter:
            continue
        selected.append(method)
    return tuple(selected)


def main_method_sort_index(family: str, window: int | None) -> int:
    method = main_method_for_family_window(family, window)
    if method is None:
        return len(MAIN_METHOD_SPECS)
    return MAIN_METHOD_ORDER[method.key]

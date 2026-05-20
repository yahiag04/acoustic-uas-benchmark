from __future__ import annotations


def normalize_label(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"1", "drone", "uav", "uas"}:
        return "drone"
    if text in {"0", "no_drone", "nodrone", "non_drone", "unknown", "background"}:
        return "no_drone"
    raise ValueError(f"Unsupported label value: {value!r}")

"""
config_loader.py — JSON loading and field-access utilities.

Mirrors the helper functions in tools/Matlab/run_topopt_from_json.m:
  reqNum, reqInt, reqStr, hasFieldPath, getFieldPath.
"""

from __future__ import annotations
import json
import os
from typing import Any


def load_json_config(json_input: str | dict) -> dict:
    """Load a topology-optimization JSON config.

    Parameters
    ----------
    json_input : str or dict
        Path to a JSON file, or an already-decoded dict.

    Returns
    -------
    dict
        The decoded configuration.
    """
    if isinstance(json_input, dict):
        return json_input
    if not isinstance(json_input, str):
        raise TypeError(f"json_input must be a file path (str) or dict, got {type(json_input)}")
    if not os.path.isfile(json_input):
        raise FileNotFoundError(f"JSON file not found: {json_input}")
    with open(json_input, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Nested field access helpers (mirrors hasFieldPath / getFieldPath)
# ---------------------------------------------------------------------------

def has_field_path(cfg: dict, path: list[str]) -> bool:
    """Return True if nested key path exists in cfg."""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def get_field_path(cfg: dict, path: list[str]) -> Any:
    """Return value at nested key path; raises KeyError if missing."""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing key '{'.'.join(path)}' in config")
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Required-field extractors with descriptive errors
# ---------------------------------------------------------------------------

def req_num(cfg: dict, path: list[str], label: str) -> float:
    """Get a required numeric scalar from cfg at nested path."""
    if not has_field_path(cfg, path):
        raise KeyError(f"Required numeric field '{label}' is missing from config.")
    val = get_field_path(cfg, path)
    if val is None:
        raise ValueError(f"Required numeric field '{label}' is null.")
    try:
        return float(val)
    except (TypeError, ValueError):
        raise ValueError(f"'{label}' must be a numeric scalar (got {val!r}).")


def req_int(cfg: dict, path: list[str], label: str) -> int:
    """Get a required integer from cfg at nested path."""
    v = req_num(cfg, path, label)
    iv = int(round(v))
    if abs(v - iv) > 1e-9:
        raise ValueError(f"'{label}' must be an integer (got {v!r}).")
    return iv


def req_str(cfg: dict, path: list[str], label: str) -> str:
    """Get a required string from cfg at nested path."""
    if not has_field_path(cfg, path):
        raise KeyError(f"Required string field '{label}' is missing from config.")
    val = get_field_path(cfg, path)
    if val is None:
        raise ValueError(f"Required string field '{label}' is null.")
    if not isinstance(val, str):
        raise ValueError(f"'{label}' must be a string (got {val!r}).")
    return val


def parse_bool(val: Any, label: str) -> bool:
    """Parse a bool-like value (bool, int, or string)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if isinstance(val, str):
        lv = val.strip().lower()
        if lv in {"true", "yes", "1", "on"}:
            return True
        if lv in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"'{label}' must be boolean-like (got {val!r}).")

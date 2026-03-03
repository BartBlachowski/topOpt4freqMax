"""
load_cases.py — Validate and normalize domain.load_cases JSON entries.

Mirrors tools/Matlab/validateLoadCases.m.

Normalized output format (list of dicts):
    [{
        "name":   str,
        "factor": float >= 0,
        "loads":  [
            {
                "type":         "self_weight" | "closest_node" | "harmonic" | "semi_harmonic",
                "factor":       float,        # required for self_weight; optional otherwise
                "location":     [x, y],       # closest_node only
                "force":        [Fx, Fy],     # closest_node only
                "mode":         int >= 1,     # harmonic / semi_harmonic only
                "update_after": int >= 0,     # harmonic only; default 1
            },
            ...
        ],
    }, ...]
"""

from __future__ import annotations
from typing import Any


def validate_load_cases(
    load_cases_raw: list | dict,
    base_path: str = "domain.load_cases",
) -> list[dict]:
    """Validate and normalize load cases from JSON.

    Parameters
    ----------
    load_cases_raw : list or dict
        Raw value of the ``domain.load_cases`` JSON field.
    base_path : str
        Label used in error messages.

    Returns
    -------
    list of dict
        Normalized load-case list.
    """
    cases = _normalize_list(load_cases_raw, base_path)
    if not cases:
        raise ValueError(f"{base_path} must contain at least one case.")

    result = []
    for i, c in enumerate(cases):
        case_path = f"{base_path}[{i}]"

        # Name
        case_name = c.get("name", f"case{i + 1}")
        if not isinstance(case_name, str):
            raise ValueError(f"{case_path}.name must be a string.")

        # Factor
        case_factor = 1.0
        if "factor" in c and c["factor"] is not None:
            cf = float(c["factor"])
            if cf < 0:
                raise ValueError(f"{case_path}.factor must be >= 0.")
            case_factor = cf

        # Loads
        loads_raw = c.get("loads", None)
        if not loads_raw:
            raise ValueError(f"{case_path}.loads must be a non-empty array.")
        loads_list = _normalize_list(loads_raw, f"{case_path}.loads")
        if not loads_list:
            raise ValueError(f"{case_path}.loads must contain at least one load.")

        loads_norm = []
        for j, ld in enumerate(loads_list):
            load_path = f"{case_path}.loads[{j}]"
            load_type = ld.get("type", None)
            if not isinstance(load_type, str):
                raise ValueError(f"{load_path}.type is required and must be a string.")
            load_type = load_type.strip().lower()

            norm = {"type": load_type}

            if load_type == "self_weight":
                if "factor" not in ld or ld["factor"] is None:
                    raise ValueError(f"{load_path}.factor is required for self_weight.")
                norm["factor"] = float(ld["factor"])

            elif load_type == "closest_node":
                norm["factor"] = float(ld.get("factor", 1.0))
                if "location" not in ld:
                    raise ValueError(f"{load_path}.location is required for closest_node.")
                loc = list(ld["location"])
                if len(loc) != 2:
                    raise ValueError(f"{load_path}.location must be [x, y].")
                norm["location"] = [float(loc[0]), float(loc[1])]
                if "force" not in ld:
                    raise ValueError(f"{load_path}.force is required for closest_node.")
                frc = list(ld["force"])
                if len(frc) != 2:
                    raise ValueError(f"{load_path}.force must be [Fx, Fy].")
                norm["force"] = [float(frc[0]), float(frc[1])]

            elif load_type == "harmonic":
                norm["factor"] = float(ld.get("factor", 1.0))
                if "mode" not in ld:
                    raise ValueError(f"{load_path}.mode is required for harmonic.")
                mode = int(round(float(ld["mode"])))
                if mode < 1:
                    raise ValueError(f"{load_path}.mode must be an integer >= 1.")
                norm["mode"] = mode
                update_after = 1
                if "update_after" in ld and ld["update_after"] is not None:
                    ua = int(round(float(ld["update_after"])))
                    if ua < 0:
                        raise ValueError(f"{load_path}.update_after must be a nonneg integer.")
                    update_after = ua
                norm["update_after"] = update_after

            elif load_type == "semi_harmonic":
                norm["factor"] = float(ld.get("factor", 1.0))
                if "mode" not in ld:
                    raise ValueError(f"{load_path}.mode is required for semi_harmonic.")
                mode = int(round(float(ld["mode"])))
                if mode < 1:
                    raise ValueError(f"{load_path}.mode must be an integer >= 1.")
                norm["mode"] = mode

            else:
                raise ValueError(
                    f'{load_path}.type="{load_type}" is not supported. '
                    "Supported types: self_weight, closest_node, harmonic, semi_harmonic."
                )

            loads_norm.append(norm)

        result.append({
            "name": case_name,
            "factor": case_factor,
            "loads": loads_norm,
        })

    return result


# ---------------------------------------------------------------------------
def _normalize_list(raw: Any, label: str) -> list:
    """Normalize a JSON value that should be a list of objects."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    raise ValueError(f"{label} must be a JSON object array.")

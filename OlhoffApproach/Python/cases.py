from __future__ import annotations

from .solver import OlhoffConfig


PAPER_TARGETS = {
    "CC": {"init": 146.1, "opt": 456.4},
    "CS": {"init": 104.1, "opt": 288.7},
    "SS": {"init": 68.7, "opt": 174.7},
}


def case_config(case: str, quick: bool = False, seed: int | None = None) -> OlhoffConfig:
    case = case.upper()
    if case not in {"CC", "CS", "SS"}:
        raise ValueError("case must be one of CC, CS, SS")

    cfg = OlhoffConfig()
    cfg.support_type = case
    cfg.seed = seed

    if quick:
        cfg.nelx = 80
        cfg.nely = 10
        cfg.maxiter = 18
        cfg.beta_interval = 8
        cfg.n_polish = 8
        cfg.rmin = 2 * cfg.L / cfg.nelx
    else:
        cfg.nelx = 240
        cfg.nely = 30
        cfg.maxiter = 300
        cfg.rmin = 2 * cfg.L / cfg.nelx

    cfg.finalize()
    return cfg

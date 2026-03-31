from __future__ import annotations


CASES = {
    "fig4": {
        "nelx": 320,
        "nely": 40,
        "volfrac": 0.5,
        "penal": 3.0,
        "rmin": 2.5,
        "ft": 1,
        "ftBC": "N",
        "eta": 0.5,
        "beta": 1.0,
        "move": 0.2,
        "maxit": 200,
        "stage1_maxit": 200,
        "bcType": "simply",
        "dyn_move": 0.01,
        "dyn_maxit": 200,
    },
    "fig8": {
        "nelx": 320,
        "nely": 40,
        "volfrac": 0.5,
        "penal": 3.0,
        "rmin": 2.0,
        "ft": 1,
        "ftBC": "N",
        "eta": 0.5,
        "beta": 1.0,
        "move": 0.2,
        "maxit": 400,
        "stage1_maxit": 200,
        "bcType": "fixedPinned",
        "dyn_move": 0.01,
        "dyn_maxit": 200,
    },
    "fig9": {
        "nelx": 150,
        "nely": 100,
        "volfrac": 0.5,
        "penal": 3.0,
        "rmin": 2.3,
        "ft": 1,
        "ftBC": "N",
        "eta": 0.5,
        "beta": 1.0,
        "move": 0.2,
        "maxit": 200,
        "stage1_maxit": 200,
        "bcType": "cantilever",
        "dyn_move": 0.01,
        "dyn_maxit": 200,
    },
}


def get_case(case: str, quick: bool = False) -> dict:
    if case not in CASES:
        raise ValueError(f"Unknown case '{case}'. Available: {', '.join(CASES)}")
    cfg = dict(CASES[case])

    if quick:
        if case == "fig9":
            cfg["nelx"] = 60
            cfg["nely"] = 40
        else:
            cfg["nelx"] = 96
            cfg["nely"] = 12
        cfg["maxit"] = 20
        cfg["stage1_maxit"] = 12
        cfg["dyn_maxit"] = 20

    return cfg

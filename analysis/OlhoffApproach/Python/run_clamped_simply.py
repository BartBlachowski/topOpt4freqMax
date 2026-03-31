#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .run_case import run_case
except ImportError:  # direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from OlhoffApproach.Python.run_case import run_case


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Olhoff clamped-simply (CS) benchmark.")
    parser.add_argument("--quick", action="store_true", help="Reduced mesh/iterations for smoke tests")
    parser.add_argument("--out", type=Path, default=None, help="Output base folder")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--snapshot-every", type=int, default=5, help="Snapshot frequency (0 disables)")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    outputs = run_case(
        case="CS",
        quick=args.quick,
        out=args.out,
        max_iter=args.max_iter,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
    )
    for p in outputs:
        print(f"Saved outputs in: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

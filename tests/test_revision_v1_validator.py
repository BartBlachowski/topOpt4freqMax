"""
tests/test_revision_v1_validator.py

Unit tests for V1c schema and validator cleanup.

Tests cover:
  1. check_gate_a0_constraints() rejects each obsolete/incompatible key.
  2. Gate A0 fixture passes check_gate_a0_constraints() without error.
  3. Configs WITHOUT gate_a0_diagnostics are NOT rejected (backward compat).
  4. validate_load_cases() correctly handles revision_v1-style load cases.
  5. run_topopt_from_json() raises ValueError for bad gate_a0 configs before
     the solver is ever imported.

Run with:
    python -m unittest tests/test_revision_v1_validator.py -v
or from repo root:
    python -m pytest tests/test_revision_v1_validator.py -v   (if pytest installed)
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so tools.* packages resolve.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.Python.load_cases import validate_load_cases
from tools.Python.run_topopt_from_json import check_gate_a0_constraints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gate_a0_fixture() -> dict:
    """Return the decoded Gate A0 fixture dict."""
    fixture_path = _REPO / "scripts" / "revision_v1" / "gate_a0_fixture.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _base_gate_a0_opt() -> dict:
    """Minimal optimization block that satisfies all Gate A0 constraints."""
    return {
        "approach": "ourApproach",
        "semi_harmonic_baseline": "solid",
        "load_sensitivity": "omitted",
        # harmonic_normalize absent → not checked (only rejected when explicitly true)
    }


def _cfg_with(opt_overrides: dict) -> dict:
    """Return a minimal cfg dict for check_gate_a0_constraints testing."""
    opt = {**_base_gate_a0_opt(), **opt_overrides}
    return {"optimization": opt}


# ===========================================================================
# 1. check_gate_a0_constraints — rejection cases
# ===========================================================================

class TestCheckGateA0ConstraintsRejection(unittest.TestCase):

    def test_rejects_semi_harmonic_rho_source(self):
        """semi_harmonic_rho_source must be absent for Gate A0."""
        cfg = _cfg_with({"semi_harmonic_rho_source": "x"})
        with self.assertRaises(ValueError) as cm:
            check_gate_a0_constraints(cfg)
        self.assertIn("semi_harmonic_rho_source", str(cm.exception))

    def test_rejects_semi_harmonic_rho_source_x_filtered(self):
        cfg = _cfg_with({"semi_harmonic_rho_source": "x_filtered"})
        with self.assertRaises(ValueError) as cm:
            check_gate_a0_constraints(cfg)
        self.assertIn("semi_harmonic_rho_source", str(cm.exception))

    def test_rejects_harmonic_normalize_true(self):
        """harmonic_normalize=true must be rejected."""
        cfg = _cfg_with({"harmonic_normalize": True})
        with self.assertRaises(ValueError) as cm:
            check_gate_a0_constraints(cfg)
        self.assertIn("harmonic_normalize", str(cm.exception))

    def test_rejects_non_solid_baseline_initial(self):
        """semi_harmonic_baseline='initial' is not solid → rejected."""
        cfg = _cfg_with({"semi_harmonic_baseline": "initial"})
        with self.assertRaises(ValueError) as cm:
            check_gate_a0_constraints(cfg)
        self.assertIn("semi_harmonic_baseline", str(cm.exception))
        self.assertIn("solid", str(cm.exception))

    def test_rejects_non_solid_baseline_current(self):
        """semi_harmonic_baseline='current' is not solid → rejected."""
        cfg = _cfg_with({"semi_harmonic_baseline": "current"})
        with self.assertRaises(ValueError) as cm:
            check_gate_a0_constraints(cfg)
        self.assertIn("semi_harmonic_baseline", str(cm.exception))

    def test_rejects_missing_load_sensitivity(self):
        """load_sensitivity must be explicitly present."""
        opt = {k: v for k, v in _base_gate_a0_opt().items() if k != "load_sensitivity"}
        cfg = {"optimization": opt}
        with self.assertRaises(ValueError) as cm:
            check_gate_a0_constraints(cfg)
        self.assertIn("load_sensitivity", str(cm.exception))

    def test_rejects_rho_source_even_when_baseline_solid(self):
        """Checks are independent: rho_source still rejected even with solid baseline."""
        cfg = _cfg_with({
            "semi_harmonic_rho_source": "x",
            "semi_harmonic_baseline": "solid",
        })
        with self.assertRaises(ValueError) as cm:
            check_gate_a0_constraints(cfg)
        self.assertIn("semi_harmonic_rho_source", str(cm.exception))


# ===========================================================================
# 2. check_gate_a0_constraints — valid / acceptance cases
# ===========================================================================

class TestCheckGateA0ConstraintsAcceptance(unittest.TestCase):

    def test_gate_a0_fixture_passes(self):
        """The actual Gate A0 fixture must satisfy all constraints without error."""
        cfg = _load_gate_a0_fixture()
        try:
            check_gate_a0_constraints(cfg)
        except ValueError as exc:
            self.fail(f"Gate A0 fixture raised ValueError unexpectedly: {exc}")

    def test_valid_omitted_sensitivity(self):
        cfg = _cfg_with({"load_sensitivity": "omitted"})
        check_gate_a0_constraints(cfg)  # should not raise

    def test_valid_complete_sensitivity(self):
        cfg = _cfg_with({"load_sensitivity": "complete"})
        check_gate_a0_constraints(cfg)  # should not raise

    def test_harmonic_normalize_false_is_accepted(self):
        """Explicit harmonic_normalize=false is fine."""
        cfg = _cfg_with({"harmonic_normalize": False})
        check_gate_a0_constraints(cfg)  # should not raise

    def test_harmonic_normalize_absent_is_accepted(self):
        """Absent harmonic_normalize does not trigger the check."""
        opt = {k: v for k, v in _base_gate_a0_opt().items() if k != "harmonic_normalize"}
        opt.pop("harmonic_normalize", None)
        cfg = {"optimization": opt}
        check_gate_a0_constraints(cfg)  # should not raise

    def test_semi_harmonic_baseline_absent_is_accepted(self):
        """Absent semi_harmonic_baseline does not trigger the baseline check."""
        opt = {k: v for k, v in _base_gate_a0_opt().items()
               if k != "semi_harmonic_baseline"}
        cfg = {"optimization": opt}
        check_gate_a0_constraints(cfg)  # should not raise


# ===========================================================================
# 3. Backward compatibility — configs WITHOUT gate_a0_diagnostics
# ===========================================================================

class TestBackwardCompatibility(unittest.TestCase):
    """Configs that do NOT set gate_a0_diagnostics must not be rejected by
    check_gate_a0_constraints (which is only called when the flag is true)."""

    def test_rho_source_and_normalize_without_flag_not_rejected_by_gate_a0(self):
        """Without gate_a0_diagnostics=true the gate_a0 guard is never invoked.

        semi_harmonic_rho_source and harmonic_normalize=true are only rejected by
        check_gate_a0_constraints, which is only called when gate_a0_diagnostics=true.
        A legacy config with these keys but without the flag must not raise ValueError
        from the V1c gate_a0 guard.

        Note: the solver already unconditionally requires semi_harmonic_baseline="solid"
        for semi_harmonic loads (pre-existing constraint, not part of V1c). The test
        uses "solid" baseline to isolate the V1c gate_a0 guard behavior.
        """
        # Test check_gate_a0_constraints directly: calling it on a config WITHOUT
        # gate_a0_diagnostics means the caller would never invoke it. But we can
        # verify it by NOT calling it and checking no ValueError in run_topopt_from_json.
        from tools.Python.run_topopt_from_json import run_topopt_from_json

        legacy_cfg = {
            "meta": {"name": "legacy compat test"},
            "domain": {
                "shape": "rectangular",
                "size": {"length": 1.0, "height": 1.0},
                "thickness": 1.0,
                "mesh": {"nelx": 2, "nely": 1},
                "load_cases": [
                    {"name": "c1", "factor": 1.0,
                     "loads": [{"type": "semi_harmonic", "mode": 1}]}
                ],
            },
            "material": {"model": "plane_stress_isotropic",
                         "E": 1e7, "nu": 0.3, "rho": 1.0},
            "void_material": {"E_min_ratio": 1e-6, "rho_min": 1e-6},
            "bc": {"supports": [
                {"type": "vertical_line", "x": 0.0, "dofs": ["ux", "uy"], "tol": 1e-9}
            ]},
            "optimization": {
                "approach": "ourApproach",
                "optimizer": "OC",
                "objective": "compliance",
                "interpolation": "SIMP",
                "volume_fraction": 0.5,
                "penalization": 3.0,
                # V1c gate_a0-rejected keys present WITHOUT gate_a0_diagnostics:
                # must NOT raise ValueError from check_gate_a0_constraints.
                # (The solver's own pre-V1c check requires solid baseline, so we use it.)
                "semi_harmonic_baseline": "solid",
                "semi_harmonic_rho_source": "x",
                "harmonic_normalize": True,
                # No gate_a0_diagnostics key.
                "filter": {"type": "sensitivity", "radius": 1.0,
                           "radius_units": "element", "boundary_condition": "symmetric"},
                "move_limit": 0.2,
                "max_iters": 1,
                "convergence_tol": 1e-12,
                "seed": 0,
            },
            "postprocessing": {"visualize_live": False},
        }

        # Must NOT raise ValueError from the gate_a0 guard (check_gate_a0_constraints).
        # The solver may raise ValueError for other reasons (e.g., semi_harmonic_rho_source
        # compatibility inside the solver), or complete successfully — both are acceptable
        # for this backward-compat test.  Only a ValueError that mentions a gate_a0
        # constraint should be treated as a failure.
        gate_a0_keywords = {
            "gate_a0", "revision_v1", "authoritative",
            "semi_harmonic_rho_source", "harmonic_normalize",
        }
        try:
            run_topopt_from_json(legacy_cfg)
        except ValueError as exc:
            msg = str(exc).lower()
            # If the error mentions gate_a0-specific language, that is unexpected.
            for kw in gate_a0_keywords:
                if kw.lower() in msg and "gate_a0_diagnostics" not in legacy_cfg.get(
                    "optimization", {}
                ):
                    # The key "semi_harmonic_rho_source" also appears in solver messages;
                    # distinguish gate_a0 guard errors by the phrase "gate a0" or
                    # "revision_v1" which our new guard always includes.
                    if "gate a0" in msg or "revision_v1" in msg:
                        self.fail(
                            f"Legacy config (no gate_a0_diagnostics) raised a gate_a0 "
                            f"guard ValueError unexpectedly: {exc}"
                        )
        except Exception:
            # ImportError, RuntimeError, etc. mean we passed the validator — acceptable.
            pass


# ===========================================================================
# 4. run_topopt_from_json raises ValueError BEFORE solver for bad gate_a0 configs
# ===========================================================================

class TestRunTopoptRaisesBeforeSolver(unittest.TestCase):
    """Integration tests: bad gate_a0 configs raise ValueError (not ImportError etc.)."""

    def _make_gate_a0_cfg(self, **opt_overrides) -> dict:
        """Gate A0–style config with gate_a0_diagnostics=true and overrides applied."""
        cfg = _load_gate_a0_fixture()
        cfg = copy.deepcopy(cfg)
        cfg["optimization"].update(opt_overrides)
        return cfg

    def _assert_raises_value_error(self, cfg: dict, keyword: str) -> None:
        from tools.Python.run_topopt_from_json import run_topopt_from_json
        with self.assertRaises(ValueError) as cm:
            run_topopt_from_json(cfg)
        self.assertIn(keyword, str(cm.exception),
                      f"Expected '{keyword}' in error message: {cm.exception}")

    def test_rho_source_raises_before_solver(self):
        cfg = self._make_gate_a0_cfg(semi_harmonic_rho_source="x")
        self._assert_raises_value_error(cfg, "semi_harmonic_rho_source")

    def test_harmonic_normalize_true_raises_before_solver(self):
        cfg = self._make_gate_a0_cfg(harmonic_normalize=True)
        self._assert_raises_value_error(cfg, "harmonic_normalize")

    def test_non_solid_baseline_raises_before_solver(self):
        cfg = self._make_gate_a0_cfg(semi_harmonic_baseline="current")
        self._assert_raises_value_error(cfg, "semi_harmonic_baseline")

    def test_missing_load_sensitivity_raises_before_solver(self):
        cfg = self._make_gate_a0_cfg()
        del cfg["optimization"]["load_sensitivity"]
        self._assert_raises_value_error(cfg, "load_sensitivity")


# ===========================================================================
# 5. validate_load_cases — revision_v1 load-case structure
# ===========================================================================

class TestValidateLoadCasesRevisionV1(unittest.TestCase):
    """Gate A0 fixture and revision_v1 style load cases parse correctly."""

    def test_gate_a0_fixture_load_cases(self):
        """Gate A0 fixture load case (single semi_harmonic, mode=1) validates."""
        fixture = _load_gate_a0_fixture()
        raw = fixture["domain"]["load_cases"]
        result = validate_load_cases(raw, "domain.load_cases")
        self.assertEqual(len(result), 1)
        lc = result[0]
        self.assertEqual(lc["name"], "reference_mode_1")
        self.assertAlmostEqual(lc["factor"], 1.0)
        self.assertEqual(len(lc["loads"]), 1)
        ld = lc["loads"][0]
        self.assertEqual(ld["type"], "semi_harmonic")
        self.assertEqual(ld["mode"], 1)

    def test_two_mode_load_cases(self):
        """Two-mode weighted combination validates (as used in clamped beam configs)."""
        raw = [
            {"name": "mode1", "factor": 1.0,
             "loads": [{"type": "semi_harmonic", "mode": 1, "factor": 1.0}]},
            {"name": "mode2", "factor": 0.0,
             "loads": [{"type": "semi_harmonic", "mode": 2, "factor": 1.0}]},
        ]
        result = validate_load_cases(raw, "domain.load_cases")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["loads"][0]["mode"], 1)
        self.assertEqual(result[1]["loads"][0]["mode"], 2)
        self.assertAlmostEqual(result[1]["factor"], 0.0)

    def test_missing_mode_raises(self):
        """semi_harmonic load without mode field is rejected."""
        raw = [{"name": "c1", "factor": 1.0,
                "loads": [{"type": "semi_harmonic"}]}]
        with self.assertRaises(ValueError) as cm:
            validate_load_cases(raw, "domain.load_cases")
        self.assertIn("mode", str(cm.exception))

    def test_invalid_load_type_raises(self):
        """Unknown load type is rejected."""
        raw = [{"name": "c1", "factor": 1.0,
                "loads": [{"type": "projected_density_rho_source", "mode": 1}]}]
        with self.assertRaises(ValueError):
            validate_load_cases(raw, "domain.load_cases")


if __name__ == "__main__":
    unittest.main()

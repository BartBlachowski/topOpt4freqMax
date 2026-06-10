# AI Writing Workflow

This document defines the execution pipelines for generating, validating, and refining a scientific paper using the ai/ system.

The workflow is dependency-aware and artifact-driven. Each stage must produce outputs in ai/out/.

---

# 1. ABSTRACT PIPELINE (default entry point)

Steps:

1. abstract_audit
2. abstract_filter
3. abstract_rewrite
4. jargon_detector
5. abstract_score

Decision:

- Score ≥ 90 → DONE
- 80–89 → repeat from step 2
- < 80 → major rewrite required

---

# 2. LITERATURE PIPELINE (prerequisite)

Required for:
- Introduction
- Related Work
- Positioning

Steps:

1. literature_extract
2. literature_balance

Outputs:

- ai/out/literature/literature_extract.md
- ai/out/literature/literature_balance.md

---

# 3. INTRODUCTION PIPELINE

Use when:
- paper draft exists
- literature outputs exist
- introduction is missing or weak

Steps:

1. write_introduction
2. gap_stress_test
3. introduction_audit
4. issue_filter
5. safe_patch
6. reviewer_simulation

Decision:

- READY → proceed
- BORDERLINE → accept with known risks
- NOT READY → fix literature grounding or gap logic

---

# 4. RELATED WORK PIPELINE

Use when:
- literature_extract exists
- citations need structuring or improvement

Steps:

1. related_work_writer
2. literature_balance (re-check)
3. reviewer_simulation

Decision:

- COMPLETE → proceed
- IMBALANCED → refine coverage
- WEAK → rewrite argumentation

---

# 5. METHODOLOGY PIPELINE

Use when:
- implementation exists
- method description is required or untrusted

Steps:

1. methodology_writer
2. section_audit
3. cross_consistency
4. issue_filter
5. safe_patch

Rules:

- Equations describe the physical/numerical problem (not library internals)
- Implementation details are secondary
- No unexplained algorithmic steps

Decision:

- VALID → proceed
- INCONSISTENT → fix alignment with code/results

---

# 6. RESULTS PIPELINE

If no results manifest exists:
- run results_discovery first
- review and promote suggested manifest to ai/config/results_manifest.json
- then run results_manifest_validator

Use when:
- experiments are completed
- results section needs validation

Steps:

1. results_writer
2. results_validator
3. cross_consistency

Rules:

- All claims must be supported by data
- No metric without definition
- No qualitative claim without quantitative backing

Decision:

- VALIDATED → proceed
- WEAK → add evidence or reduce claims



---

# 7. POSITIONING PIPELINE

Purpose:
- refine contributions
- establish novelty

Steps:

1. contribution_refiner
2. novelty_positioning
3. gap_stress_test (final check)

Decision:

- CLEAR → proceed
- OVERCLAIM → reduce claims
- UNCLEAR → refine contributions

---

# 8 Numerical results pipeline

Use when:
- numerical outputs exist
- ai/config/results_manifest.json exists

Steps:
1. results_manifest_validator
2. write_numerical_results
3. results_validator
4. issue_filter
5. safe_patch

Decision:
- VALIDATED -> proceed
- INVALID -> fix manifest
- WEAK -> reduce claims or add evidence

# 9. CONCLUSION PIPELINE

Prerequisites:

- results_validator
- contribution_refiner
- novelty_positioning

Steps:

1. conclusion_writer
2. conclusion_audit

Rules:

- No new claims
- Must reflect validated results
- Must align with contributions

Decision:

- VALID → proceed
- INFLATED → reduce claims
- WEAK → strengthen synthesis

---

# 10. FULL PAPER PIPELINE (orchestrator)

Use:

- ai/run/run_full_paper_pipeline.md

Responsibilities:

- check all required artifacts
- detect BLOCKED states
- stop if dependencies missing
- produce final_status.md

Requires:

- Abstract ≥ 90
- Introduction READY
- Methodology VALID
- Results VALIDATED
- Positioning CLEAR
- Conclusion VALID

---

# 11. AUDIT & FINALIZATION

Tools:

- agents/paper_wide_evaluator.md
- prompts/final_report_writer.md
- cross_consistency.prompt.md
- final_submission_guard.md
- journal_selector.md

---

# EXECUTION PRINCIPLES

1. All stages must write outputs to ai/out/ on local machine
2. Inline-only execution is invalid
3. A stage is complete only if artifacts exist
4. Downstream stages must not guess missing inputs
5. BLOCKED state is correct behavior, not failure

---

## NON-NEGOTIABLE EXECUTION RULE

All stages MUST write their declared outputs to the ai/out/ directory on the local machine.

The filesystem is the single source of truth for pipeline state. This framework is designed for reproducibility, debugging, partial reruns, audit trails, modular validation, and resuming after interruption — not for one-shot inline generation.

---

## COMPLETION CRITERIA

A stage is considered COMPLETE only if:

1. The expected output file exists in ai/out/
2. The file is non-empty and contains the generated content
3. (If applicable) metadata/state files are updated consistently

---

## INVALID EXECUTION

Producing results only in the model context (inline) without writing to ai/out/ is INVALID and MUST NOT be treated as a completed stage.

---

## FAILURE HANDLING

If a stage does not write its required output file:

- status MUST be set to FAILED (not COMPLETE)
- downstream stages MUST NOT proceed
- the stage MUST be considered not executed

---

## EXECUTION VALIDATION (MANDATORY)

After each stage execution, the system MUST verify:

- Does the expected file exist in ai/out/?
- Is the file populated with content?

If verification fails → enforce FAILURE HANDLING

---

## SOURCE OF TRUTH

Pipeline correctness is determined exclusively by filesystem artifacts, not by:
- model memory
- conversation history
- inline outputs

---

# RECOVERY

If pipeline is BLOCKED:

Run:

- ai/run/run_unblock_plan.md

Then execute suggested steps.

---


# AI Writing Workflow

## Abstract pipeline (default)

1. abstract_audit
2. abstract_filter
3. abstract_rewrite
4. jargon_detector
5. abstract_score

Decision:

* Score ≥ 90 → DONE
* 80–89 → repeat from step 2
* < 80 → major rewrite required

---

## Full paper audit

Use:

* agents/paper_wide_evaluator.md

---

## When to use what

* Poor readability → jargon_detector
* Too long / detailed → abstract_filter
* Structural issues → abstract_audit
* Final decision → abstract_score

## Introduction pipeline

Use when:
- the paper draft exists
- literature_extract output exists
- literature_balance output exists
- the Introduction is missing, weak, outdated, or untrusted

Steps:
1. write_introduction
2. gap_stress_test
3. introduction_audit
4. issue_filter
5. safe_patch
6. reviewer_simulation

Decision:
- READY -> proceed
- BORDERLINE -> keep current version with explicit risks
- NOT READY -> strengthen literature grounding or gap logic first
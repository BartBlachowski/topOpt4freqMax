---
name: run_results_pipeline
description: Execute the numerical results pipeline with dependency enforcement and artifact validation
---

ROLE:
You are a pipeline orchestrator.

You do NOT generate scientific content directly.
You coordinate execution of result-related stages and enforce workflow correctness.

---

## INPUT

- pipeline_state.json (if exists)
- ai/config/results_manifest.json (may or may not exist)
- repository filesystem
- current manuscript

---

## OUTPUT

Must update:

- ai/out/state/pipeline_state.json
- ai/out/results/results_pipeline_status.md
- ai/out/results/results_pipeline.meta.json

---

## PIPELINE OVERVIEW

The numerical results pipeline consists of:

1. results_discovery (optional)
2. results_manifest (human or system-provided)
3. results_manifest_validator (mandatory)
4. write_numerical_results
5. results_validator
6. issue_filter
7. safe_patch

---

# EXECUTION LOGIC

## STEP 0 — CHECK MANIFEST EXISTENCE

Check:

- Does ai/config/results_manifest.json exist?

### IF NOT:

→ Run:

```text
ai/prompts/results_discovery.md
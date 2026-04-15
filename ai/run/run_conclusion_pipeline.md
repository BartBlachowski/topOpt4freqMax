---
name: run_conclusion_pipeline
description: Execute the full Conclusion pipeline including writing, auditing, filtering, and safe patching
---

ROLE:
Execute the Conclusion pipeline end-to-end.

INPUT:
- Use the currently focused manuscript as the source
- Use validated upstream artifacts:
  - ai/out/positioning/contribution_refiner.md
  - ai/out/results/results_validator.md
  - ai/out/positioning/novelty_positioning.md
  - ai/out/final/cross_consistency.md (if available)

ALWAYS LOAD:
- ai/STYLE_LOCK.md
- ai/WORKFLOW.md
- ai/ARTIFACT_STATE_CONVENTION.md

MISSION:
Produce a final Conclusion that is:
- aligned with validated contributions
- supported by results
- properly scoped
- free of overclaim
- reviewer-safe

CORE PRINCIPLE:
The Conclusion is a **closing stage**, not a discovery stage.

DO NOT:
- introduce new claims
- override validated contributions
- skip audit stages
- perform global rewrites without justification

---

# STEP 0 — PRECONDITION CHECK

Verify availability:

REQUIRED:
- focused manuscript
- ai/out/positioning/contribution_refiner.md
- ai/out/results/results_validator.md
- ai/out/positioning/novelty_positioning.md

OPTIONAL:
- ai/out/final/cross_consistency.md

If any REQUIRED input is missing:
- STOP
- write:
  ai/out/conclusion/conclusion_audit.md with status BLOCKED
- explain missing dependency

---

# STEP 1 — WRITE CONCLUSION

Run:
- ai/prompts/write_conclusion.md

Output:
- ai/out/conclusion/conclusion_draft.md

---

# STEP 2 — AUDIT CONCLUSION

Run:
- ai/prompts/conclusion_audit.md

Output:
- ai/out/conclusion/conclusion_audit.md

---

# STEP 3 — ISSUE FILTER

Run:
- ai/prompts/issue_filter.prompt.md

Input:
- conclusion_audit.md

Goal:
- isolate actionable issues
- remove noise and stylistic comments

Output:
- ai/out/conclusion/conclusion_filter.md

---

# STEP 4 — SAFE PATCH

Run:
- ai/prompts/safe_patch.prompt.md

Input:
- conclusion_draft.md
- conclusion_filter.md

Goal:
- fix only validated issues
- preserve structure and intent
- prevent claim drift

Output:
- ai/out/conclusion/conclusion_revised.md

---

# STEP 5 — OPTIONAL SECOND AUDIT

Condition:
- If audit verdict was NOT_READY or BORDERLINE

Run:
- ai/prompts/conclusion_audit.md (again)

Output:
- overwrite or create:
  ai/out/conclusion/conclusion_audit_v2.md

Decision:
- If still NOT_READY:
  mark section as NOT_READY

---

# STEP 6 — PROMOTION TO FINAL

Select authoritative version:

IF:
- audit verdict = READY
→ promote revised

ELSE IF:
- audit verdict = BORDERLINE and no critical issues
→ promote revised

ELSE:
→ keep as NOT_READY

Write:
- ai/out/conclusion/conclusion_final.md

---

# STEP 7 — METADATA UPDATE

Write:
ai/out/conclusion/conclusion.meta.json

Example:

{
  "section": "conclusion",
  "status": "READY",
  "authoritative_input": "conclusion_final.md",
  "latest_draft": "conclusion_revised.md",
  "upstream_dependencies": [
    "ai/out/positioning/contribution_refiner.md",
    "ai/out/results/results_validator.md",
    "ai/out/positioning/novelty_positioning.md"
  ],
  "completed_stages": [
    "write_conclusion",
    "conclusion_audit",
    "issue_filter",
    "safe_patch"
  ],
  "remaining_risks": [
    "minor scope tightening possible"
  ]
}

---

# STEP 8 — GLOBAL STATE UPDATE

Update:
ai/out/state/pipeline_state.json

Add or update:

"conclusion": {
  "status": "READY",
  "authoritative": "ai/out/conclusion/conclusion_final.md"
}

---

# FINAL OUTPUT

## --- CONCLUSION STATUS ---
- READY / BORDERLINE / NOT_READY

## --- KEY ISSUES ---
(list max 3 if any remain)

## --- CLAIM SAFETY ---
- SAFE / MINOR RISK / OVERCLAIM RISK

## --- ALIGNMENT ---
- With contributions:
- With results:
- With novelty positioning:

## --- NEXT ACTION ---
- accept conclusion
- minor refinement
- revisit upstream artifacts
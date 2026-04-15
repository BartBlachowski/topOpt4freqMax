---
name: run_related_work_pipeline
description: Execute the full Related Work pipeline using structured literature extraction and balance control
---

ROLE:
Execute the Related Work pipeline end-to-end.

INPUT:
- Use the currently focused manuscript as the source paper
- Use:
  - ai/out/literature/literature_extract.md
  - ai/out/literature/literature_balance.md
- Use the manuscript only for alignment, not as the main literature source

ALWAYS LOAD:
- ai/STYLE_LOCK.md
- ai/WORKFLOW.md
- ai/ARTIFACT_STATE_CONVENTION.md

MISSION:
Produce a Related Work section that is:
- synthesis-driven
- balanced
- literature-grounded
- non-redundant
- reviewer-safe

CORE PRINCIPLE:
Related Work must compare and group prior work.
It must not become a citation dump or duplicate the Introduction.

DO NOT:
- write from memory when literature extraction exists
- summarize papers one by one
- overstate limitations not supported by extraction
- duplicate the Introduction gap paragraph
- skip audit and patch stages when issues are present

---

# STEP 0 — PRECONDITION CHECK

Verify availability:

REQUIRED:
- focused manuscript
- ai/out/literature/literature_extract.md
- ai/out/literature/literature_balance.md

If any REQUIRED input is missing:
- STOP
- write:
  ai/out/related_work/related_work_audit.md with status BLOCKED
- explain missing dependency

---

# STEP 1 — WRITE RELATED WORK

Run:
- ai/prompts/related_work_writer.md

Output:
- ai/out/related_work/related_work_draft.md

---

# STEP 2 — AUDIT RELATED WORK

Run:
- ai/prompts/section_audit.prompt.md

Focus the audit on:
- synthesis vs listing
- grouping quality
- unsupported critique
- duplication with Introduction
- weak transitions
- citation overload
- imbalance or omission

Output:
- ai/out/related_work/related_work_audit.md

---

# STEP 3 — ISSUE FILTER

Run:
- ai/prompts/issue_filter.prompt.md

Input:
- related_work_audit.md

Goal:
- classify issues into:
  - ESSENTIAL
  - OPTIONAL
  - HARMFUL TO CHANGE

Output:
- ai/out/related_work/related_work_filter.md

---

# STEP 4 — SAFE PATCH

Run:
- ai/prompts/safe_patch.prompt.md

Input:
- related_work_draft.md
- related_work_filter.md

Goal:
- apply only justified fixes
- preserve grouping logic
- prevent over-editing
- avoid turning synthesis into generic prose

Output:
- ai/out/related_work/related_work_revised.md

---

# STEP 5 — OPTIONAL SECOND AUDIT

Condition:
- If first audit verdict was BORDERLINE or NOT_READY
- or if safe patch materially changed section structure

Run again:
- ai/prompts/section_audit.prompt.md

Output:
- ai/out/related_work/related_work_audit_v2.md

---

# STEP 6 — PROMOTION TO FINAL

Promote authoritative version:

IF:
- audit verdict = READY
→ promote revised

ELSE IF:
- audit verdict = BORDERLINE and no critical issues remain
→ promote revised

ELSE:
→ keep as NOT_READY

Write:
- ai/out/related_work/related_work_final.md

---

# STEP 7 — METADATA UPDATE

Write:
- ai/out/related_work/related_work.meta.json

Example:

{
  "section": "related_work",
  "status": "READY",
  "authoritative_input": "related_work_final.md",
  "latest_draft": "related_work_revised.md",
  "upstream_dependencies": [
    "ai/out/literature/literature_extract.md",
    "ai/out/literature/literature_balance.md"
  ],
  "completed_stages": [
    "related_work_writer",
    "section_audit",
    "issue_filter",
    "safe_patch"
  ],
  "remaining_risks": [
    "minor overlap risk with Introduction"
  ]
}

---

# STEP 8 — GLOBAL STATE UPDATE

Update:
- ai/out/state/pipeline_state.json

Add or update:

"related_work": {
  "status": "READY",
  "authoritative": "ai/out/related_work/related_work_final.md"
}

---

# FINAL OUTPUT

## --- RELATED WORK STATUS ---
- READY / BORDERLINE / NOT_READY

## --- KEY ISSUES ---
(list max 3 if any remain)

## --- BALANCE STATUS ---
- BALANCED / MINOR BIAS / MAJOR BIAS

## --- SYNTHESIS QUALITY ---
- STRONG / ADEQUATE / WEAK

## --- DUPLICATION RISK ---
- LOW / MEDIUM / HIGH

## --- NEXT ACTION ---
- accept related work
- minor refinement
- strengthen literature extraction
- rebalance literature coverage
Load and follow:

1. ai/STYLE_LOCK.md
2. ai/prompts/issue_filter.prompt.md
3. ai/prompts/safe_patch.prompt.md

INPUT:
Use the most recent audit already produced in this conversation.

TASK:

STEP 1 — Extract issues from the audit

STEP 2 — Classify each issue:
- ESSENTIAL (must fix)
- OPTIONAL (nice to improve)
- HARMFUL (do not change)

STEP 3 — Build SAFE CHANGE SET:
Include ONLY ESSENTIAL issues

STEP 4 — Apply fixes:
- Modify ONLY necessary fragments
- Preserve structure and tone
- Do NOT rewrite entire sections
- Do NOT add new numerical details

STEP 5 — Self-check:
- Did readability improve?
- Did density increase?
- Did tone become procedural?

If any answer is negative → revise changes

OUTPUT:

--- REVISED TEXT ---
[only modified section]

--- APPLIED CHANGES ---
[what + why]

--- SKIPPED CHANGES ---
[optional or harmful items not applied]
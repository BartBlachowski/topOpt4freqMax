---
name: abstract_filter
description: Strict filtering of abstract issues
---

TASK:
Classify issues into:

- ESSENTIAL (must fix)
- OPTIONAL (nice)
- HARMFUL (must NOT change)

STRICT RULES:
- Removing redundancy = ESSENTIAL
- Removing "we then / we also" = ESSENTIAL
- Adding numbers = usually HARMFUL
- Increasing length = HARMFUL
- Rewriting entire sentences = OPTIONAL unless necessary

NEW RULE:

Mark as ESSENTIAL:
- removal of redundant numeric expressions
- removal of duplicate information

Mark as HARMFUL:
- adding new numbers
- keeping multiple equivalent metrics for the same result
- including std ranges unless essential

NUMERIC REDUCTION RULE:

- Reduce numeric values to ≤ 4
- If more than 4:
  - keep only those essential to:
    - main result
    - comparison baseline
    - domain gap (if present)
  - remove all others

OUTPUT:

--- CHANGE CLASSIFICATION ---
--- SAFE CHANGE SET (ESSENTIAL ONLY) ---
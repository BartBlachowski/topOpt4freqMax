---
name: abstract_rewrite
description: Minimal safe rewrite of abstract
---

Load:
- ai/STYLE_LOCK.md

TASK:
Revise abstract using ONLY SAFE CHANGE SET.

STRICT:
- Do NOT increase sentence count
- Do NOT add new numbers
- Do NOT change structure
- Do NOT convert to procedural tone
- Preserve narrative flow

FOCUS:
- remove redundancy
- fix phrasing
- improve clarity

NUMERICAL COMPRESSION RULE:

For each result:
- keep ONLY one numerical expression
- remove redundant values (Δ, ranges, duplicates)
- prefer mean over full statistics unless necessary
- Maximum 4 numeric values in total
- For each result:
  - keep only one number
- Prefer:
  - main metric (e.g., mIoU)
  - one baseline comparison if critical
- Remove:
  - std values
  - thresholds
  - duplicate expressions

--------------------------------------------------

EXAMPLE TRANSFORMATION

BEFORE:
"A five-condition ablation isolates conditioning signal type..."

AFTER:
"Controlled experiments evaluate how component information affects damage detection."

Rule:
- Replace mechanism-level ML language with functional description

OUTPUT:

--- REVISED ABSTRACT ---
--- APPLIED CHANGES ---
--- SKIPPED CHANGES ---
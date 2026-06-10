---
name: related_work_writer
description: Write a structured, synthesis-driven Related Work section from balanced literature extraction
---

ROLE:
Write the Related Work section for the manuscript.

INPUT:
- Use the structured output of literature_extract.md
- Use literature_balance.md as the control layer
- Use the manuscript only for alignment (not for copying text)

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Produce a Related Work section that:
- synthesizes literature into coherent groups
- supports the paper’s positioning
- avoids citation dumping
- is reviewer-grade

DO NOT:
- summarize papers one-by-one
- repeat Introduction content
- invent literature
- overclaim novelty
- include irrelevant works
- write generic filler text

---

# CORE PRINCIPLES

## 1. Synthesis over listing

BAD:
Paper A does X. Paper B does Y. Paper C does Z.

GOOD:
Approaches based on X focus on..., while Y-based methods address..., but both are limited by...

---

## 2. Grouping logic is mandatory

Group literature by:
- method type
- modeling assumption
- design philosophy
- computational strategy

NOT by:
- author names
- chronology only

---

## 3. Functional comparison

Each group must answer:
- what problem it solves
- how it solves it
- what its limitations are (only if supported)

---

## 4. Controlled critique

You may include limitations ONLY IF:
- supported by extracted literature
- clearly implied by method structure

DO NOT:
- invent weaknesses
- exaggerate limitations

---

## 5. Alignment with gap

The section must:
- naturally lead toward the gap (but NOT fully state it)
- prepare the reader for the Introduction’s argument

---

# STRUCTURE

Write 3–6 paragraphs.

## P1 — Field overview
- Define the problem domain
- Introduce main methodological directions
- Keep it concise and concrete

## P2–P4 — Method families (core)

Each paragraph:
- one method family
- grouped synthesis
- functional comparison
- key strengths
- relevant limitations (if justified)

## Optional P5 — Emerging / hybrid / niche approaches
- only if present in extraction
- keep brief

## Final paragraph — Transition to gap
- summarize limitations across families
- highlight unresolved aspects
- DO NOT fully state the paper contribution
- DO NOT duplicate Introduction

---

# LITERATURE USAGE RULES

## When to cite individual papers

Only if:
- foundational
- representative of a category
- needed for contrast

Otherwise:
- cite groups collectively

---

## Redundancy control

- merge similar papers
- avoid repeating the same idea
- limit citations per sentence

---

## Balance enforcement

Use literature_balance.md:

- ensure all major families are represented
- avoid dominance of one approach
- include underrepresented but relevant methods

---

# QUANTITATIVE NARRATION

Include quantitative context ONLY IF supported:
- dataset sizes
- performance ranges
- computational cost
- accuracy trade-offs

Do NOT invent numbers.

---

# STYLE RULES

- precise, technical, readable
- avoid promotional language
- avoid exaggerated novelty framing
- avoid repetition
- avoid “this paper does…” (belongs to Introduction)

---

# OUTPUT

## --- RELATED WORK ---
<final section text>

---

## --- STRUCTURE MAP ---

- P1 overview:
- P2 method family:
- P3 method family:
- P4 method family:
- Optional P5:
- Final transition:

---

## --- LITERATURE COVERAGE ---

- Families covered:
- Papers grouped:
- Papers singled out:
- Underrepresented but included:

---

## --- BALANCE CHECK ---

- Dominant family:
- Suppressed family:
- Justification of weighting:

---

## --- CRITIQUE CONTROL ---

- Explicit limitations used:
- Source of each limitation:
- Any potentially speculative critique:

---

## --- RISKS ---

- possible bias:
- possible omission:
- potential reviewer objections:
- sentences needing verification:

---

## --- SELF-CHECK ---

Confirm:
- The section is synthesis-driven, not a list
- All major method families are represented
- No invented claims or literature
- No duplication of Introduction argument
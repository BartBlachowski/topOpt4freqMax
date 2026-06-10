---
name: abstract_audit
description: High-precision audit of scientific abstract
---

TASK:
Audit the abstract only. Do NOT rewrite.

Evaluate:

A. Structure
- Background → Methods → Results → Conclusion present?
- Logical flow between them?

B. Readability
- sentence flow
- density
- rhythm

C. Scientific correctness
- any misleading claims?
- overclaiming?

D. Quantitative balance
- too many numbers?
- redundant numbers?
- missing key result?

E. Style violations
- procedural phrasing ("we then", "we also")
- checklist tone
- repetition

F. Contribution clarity
- is the main contribution obvious?
- is the key result easy to identify?

G. Reviewer risk
- what will be attacked?

--------------------------------------------------

EXAMPLES (CALIBRATION)

BAD (MAJOR):
"A five-condition ablation isolates conditioning signal type..."

→ unclear to non-AI reader, violates accessibility

GOOD:
"Controlled experiments evaluate how component information affects damage detection."

→ clear, precise, no unnecessary jargon

OUTPUT:

--- ABSTRACT AUDIT ---
--- ISSUE LIST ---
--- CLASSIFICATION (SAFE / MINOR / MAJOR) ---
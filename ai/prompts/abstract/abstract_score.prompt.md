TASK: Score the scientific abstract for publication readiness (0–100).

CONTEXT:

* Audience: civil/structural engineering researchers (not ML specialists)
* Journal level: Automation in Construction / Computers & Structures
* Follow ai/STYLE_LOCK.md

---

SCORING CRITERIA (total = 100)

1. CLARITY (0–20)

* Is every sentence understandable on first read?
* No ML-internal jargon (or properly translated)
* No overloaded sentences

2. ABSTRACTION LEVEL (0–20)

* No implementation details (training, variants, thresholds)
* No unnecessary statistics (std, Δ, repeated values)
* Results presented at summary level

3. NARRATIVE QUALITY (0–15)

* Logical flow: problem → method → result → conclusion
* No procedural or "paper describes" tone
* Focus on research question, not mechanism

4. RESULT COMMUNICATION (0–15)

* Key findings clearly stated
* Numbers used selectively and meaningfully
* No redundancy (same info twice)

5. DOMAIN ACCESSIBILITY (0–15)

* Understandable by non-AI domain expert
* ML concepts expressed functionally
* No unexplained terminology

6. CONTRIBUTION SHARPNESS (0–15)

* Main claim is explicit
* Null/positive result clearly stated
* No ambiguity in conclusions

PENALTIES:



---

PENALTIES (CRITICAL)

Apply automatic deductions:

* −10 each: ML-internal term (e.g., gradient coupling, soft-detach)
* −10 each: threshold notation (τ, Δ, etc.)
* −5 each: redundant numeric expression
* −5 each: sentence with ≥2 jargon phrases
* −5 each: training detail (loss functions, seeds, etc.)
* −5 each: numeric value beyond 4

---

STEP 1 — SCORE

Provide:

TOTAL SCORE: XX / 100

Breakdown per category.

---

STEP 2 — FAIL CONDITIONS

List if any of the following occur:

* unreadable sentence
* method-level detail in abstract
* more than 4 numeric values
* duplicated information

If any present → mark:

STATUS: NOT READY

Else:

STATUS: READY

---

STEP 3 — TOP 5 FIXES

List ONLY the highest-impact fixes (max 5), e.g.:

1. Remove ...
2. Rewrite ...
3. Simplify ...

---

STEP 4 — MINIMAL IMPROVED VERSION

Provide a corrected version with:

* minimal edits only
* no full rewrite unless necessary

---

STEP 5 — REVIEWER COMMENT

Write 3 sentences as a harsh reviewer explaining why this abstract would be rejected.

---

STEP 6 — CONFIDENCE

Rate confidence (LOW/MEDIUM/HIGH)

If LOW → explain uncertainty

---

GOAL:
Produce a reviewer-ready abstract with enforced abstraction discipline.

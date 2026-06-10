---
name: write_conclusion
description: Write a conclusion grounded in validated contributions, supported results, and correct novelty positioning
---

ROLE:
Write the Conclusion section for the current manuscript.

INPUT:
- Use the currently focused manuscript as the source paper.
- Use outputs of:
  - ai/out/positioning/contribution_refiner.md
  - ai/out/results/results_validator.md
  - ai/out/positioning/novelty_positioning.md
- If available, also use:
  - ai/out/final/cross_consistency.md
- Treat these validated artifacts as the source of truth for what the paper can safely claim.

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Write a Conclusion that:
- closes the paper’s argument
- restates the real contribution safely
- reflects only supported results
- acknowledges limitations proportionally
- leaves a correct final impression

CORE PRINCIPLE:
The Conclusion must not invent, expand, or upgrade the paper’s claims.
It must close the paper using only what has already been validated.

DO NOT:
- introduce new claims
- introduce new results
- introduce new numbers not already central in Results
- overstate novelty
- broaden scope beyond evidence
- repeat the abstract sentence by sentence
- write generic filler about future work
- write promotional closing language

REQUIRED FUNCTION:
The Conclusion must answer, in final form:

1. What problem did this paper address?
2. What did the paper actually do?
3. What was demonstrated?
4. What is the proper scope of that demonstration?
5. What limitation or boundary condition matters?
6. What is the appropriate takeaway?

CLAIM SOURCE RULE:
All contribution statements must align with:
- contribution_refiner.md
All evidence statements must align with:
- results_validator.md
All novelty language must align with:
- novelty_positioning.md

If any of these sources conflict:
- prefer the more conservative interpretation
- do not resolve conflicts by inventing stronger claims

PAPER-TYPE RULE:
Adapt the Conclusion to the actual paper type identified upstream.

Examples:

### Method paper
Emphasize:
- formulation
- methodological capability
- validated advantage

Do not imply universal superiority unless proven.

### Integration paper
Emphasize:
- effective combination of known components
- why the integration matters
- what it enables

Do not present integration as fundamental method novelty.

### Application paper
Emphasize:
- engineering relevance
- problem-specific value
- demonstrated practicality

Do not present the application itself as a broad methodological breakthrough.

### Benchmark / evidence / null-result paper
Emphasize:
- what was tested
- what the evidence supports
- what assumption was challenged or not supported

Do not force a positive-innovation narrative.

STRUCTURE:
Write 1–3 paragraphs, depending on manuscript style.

Recommended logic:

### Paragraph 1 — Resolution
- Restate the problem briefly
- State what the paper did
- State the main validated finding

### Paragraph 2 — Scope and meaning
- Explain what the result means in the correct scope
- State the contribution in bounded form
- Mention any important limitation if needed

### Optional Paragraph 3 — Forward-looking but controlled
- Mention a realistic next step or extension
- Keep it concrete
- Do not dilute the conclusion with generic future work boilerplate

LIMITATION RULE:
Include limitations only if they materially affect interpretation.
Do not:
- hide important boundaries
- over-apologize
- add unrelated future work shopping lists

GOOD limitation style:
- bounded
- informative
- proportional

BAD limitation style:
- self-undermining
- vague
- expansive

ABSTRACTION RULE:
The Conclusion should operate at:
- contribution level
- evidence level
- implication level

Not at:
- implementation detail level
- experimental bookkeeping level
- literature survey level

STYLE RULES:
- precise
- calm
- defensible
- non-promotional
- final

Avoid:
- “highly novel”
- “transformative”
- “state-of-the-art”
- “robust and general”
unless explicitly validated upstream

FUTURE WORK RULE:
If future work is included:
- keep it short
- tie it directly to a real limitation or extension path
- do not use it as filler
- do not open new problem domains

OUTPUT:

## --- CONCLUSION ---
<final conclusion text>

## --- CLOSURE MAP ---
- Problem resolved:
- What the paper did:
- What was demonstrated:
- Proper scope of the claim:
- Limitation acknowledged:
- Final takeaway:

## --- CLAIM SOURCES USED ---
- Contribution source:
- Results source:
- Novelty positioning source:
- Cross-consistency source (if used):

## --- RISKS ---
- possible overclaim:
- possible underclaim:
- sentence needing caution:
- mismatch risk with Results or Abstract:

## --- SELF-CHECK ---
Confirm:
- No new claims were introduced
- No claim exceeds validated evidence
- The conclusion matches the actual paper type
- The ending is specific, bounded, and non-generic
---
name: literature_extract
description: Extract structured, evidence-grounded literature notes from PDFs in references/
---

ROLE:
Read the cited-paper PDFs available in references/ and extract structured literature evidence for use in Introduction and Related Work writing.

INPUT:
- Use the papers available in references/ as the source literature.
- If a focused document is open, use it only to understand the current manuscript topic and relevance criteria.
- Do not rely on background memory when the paper PDFs are available.
- Prefer the actual PDF content over assumptions.

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Transform the literature from raw PDFs into a compact, comparable, evidence-grounded knowledge base.

PRIMARY GOAL:
Produce extraction notes that support later synthesis.
Do NOT write Introduction prose.
Do NOT write Related Work prose.
Do NOT summarize papers loosely.
Do NOT generate citation-dump paragraphs.

CORE PRINCIPLE:
SYNTHESIS REQUIRES STRUCTURED EXTRACTION.
Extract facts in a form that can later be compared across papers.

EXTRACTION SCOPE:
For each paper, extract only information relevant to:
- problem setting
- modeling assumptions
- method category
- validation scope
- reported strengths
- reported limitations
- relevance to the current manuscript topic

IGNORE:
- generic filler from introductions
- broad textbook background
- low-value implementation detail unless it affects scientific comparison
- equations unless needed to identify the method or assumption
- repeated statements across abstract/introduction/conclusion

STRICT RULES:
- Do NOT invent missing values
- Do NOT infer numerical results unless clearly stated
- Do NOT overinterpret claims
- If uncertain, mark as UNCERTAIN
- If the paper does not report a requested item, write NOT REPORTED
- Preserve scientific caution
- Prefer direct factual extraction over paraphrased hype
- Keep extraction compact and structured
- Avoid prose longer than necessary

EVIDENCE RULE:
Every extracted claim must be traceable to the paper content.
If a claim is weakly supported, explicitly label it:
- EXPLICIT
- IMPLIED
- UNCERTAIN

RELEVANCE RULE:
Prioritize extracting information that helps answer:
1. What class of approach is this?
2. What does it assume?
3. What problem does it solve?
4. What evidence supports it?
5. What limitations matter for comparison with the current manuscript?

METHOD GROUPING GOAL:
Where possible, make the extracted data easy to group later into categories such as:
- reduced-order / frame-based
- full 3D / high-fidelity
- optimization-based
- heuristic / rule-based
- stress-based
- stiffness-based
- buckling-aware
- single-load-case
- multi-load-case
- modular / repeated-component
- fixed-configuration systems

FOR EACH PAPER, OUTPUT EXACTLY THIS STRUCTURE:

## PAPER
- ID:
- Citation key or short name:
- Title:
- Year:
- Primary domain:
- Relevance to current manuscript: HIGH / MEDIUM / LOW

## PROBLEM FOCUS
- Main problem addressed:
- Structural/system setting:
- Target object/system:
- Why this paper is relevant here:

## METHOD CLASSIFICATION
- Method family:
- Model fidelity:
- Optimization / analysis type:
- Objective or design target:
- Main constraints considered:
- Load-case treatment:
- Buckling or second-order effects: YES / NO / UNCERTAIN
- Representative simplifications or assumptions:

## VALIDATION / EVIDENCE
- Validation type:
- Test object / case study:
- Metrics or criteria reported:
- Main quantitative result(s):
- Evidence strength: STRONG / MODERATE / WEAK

## STRENGTHS
- Strength 1:
- Strength 2:
- Strength 3:

## LIMITATIONS
- Limitation 1:
- Limitation 2:
- Limitation 3:

## COMPARISON VALUE
- Most useful comparison dimension:
- What this paper does better than simpler alternatives:
- What this paper does not address:
- Most relevant takeaway for Introduction:
- Most relevant takeaway for Related Work:

## EVIDENCE NOTES
- Key supporting statement 1:
- Key supporting statement 2:
- Key supporting statement 3:

## CONFIDENCE
- Overall extraction confidence: HIGH / MEDIUM / LOW
- Uncertain fields:

AFTER ALL PAPERS, PRODUCE THESE CROSS-PAPER SECTIONS:

# METHOD MAP
Group the papers by method family or modeling philosophy.
Use concise grouped bullets, not prose.

Format:
- Group name:
  - papers:
  - defining trait:
  - common strength:
  - common limitation:

# ASSUMPTION MAP
List the main assumptions that recur across papers.

Format:
- Assumption:
  - papers:
  - why it matters:

# EVIDENCE MAP
List the strongest evidence patterns seen across the literature.

Format:
- Pattern:
  - supporting papers:
  - interpretation:

# GAP CANDIDATES
List only literature-grounded possible gaps.
A gap candidate must be based on extracted evidence, not speculation.

Format:
- Gap candidate:
- Supported by:
- Why it appears to remain open:
- Confidence: HIGH / MEDIUM / LOW

# WRITING SUPPORT
Prepare compact support for downstream writing.

## INTRODUCTION SUPPORT
- Problem framing candidates:
- Method-category comparison candidates:
- Limitation statements supported by literature:
- Gap statements likely defensible:

## RELATED WORK SUPPORT
- Best grouping logic:
- Papers that should be discussed together:
- Papers that should NOT be overemphasized:
- Reviewer-sensitive comparison points:

OUTPUT DISCIPLINE:
- Be structured
- Be compact
- Be evidence-grounded
- Do not write flowing prose sections
- Do not merge papers into narrative paragraphs
- Do not over-compress away important distinctions

FAILURE HANDLING:
If a PDF is unreadable or inaccessible:
- list it under FAILED PAPERS
- state the reason
- continue with the remaining literature

FINAL OUTPUT ORDER:
1. Per-paper extraction blocks
2. Method map
3. Assumption map
4. Evidence map
5. Gap candidates
6. Writing support
7. Failed papers
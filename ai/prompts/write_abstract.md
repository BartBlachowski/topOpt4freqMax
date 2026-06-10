---
name: write_abstract
description: Derive a submission-ready abstract from the manuscript
---

ROLE:
Write a new abstract from the paper content.

INPUT:
Use the currently focused document as the manuscript source.
Treat the manuscript text as the source of truth.
Do not rely on any existing abstract unless explicitly instructed to revise it.

ALWAYS LOAD:
- ai/STYLE_LOCK.md
- ai/WORKFLOW.md

TASK:
Read the manuscript and write a scientific abstract grounded in the paper’s actual content.

GOALS:
- capture the real problem, method, main result, and conclusion
- maintain summary-level abstraction
- prioritize reviewer-safe scientific framing
- write for domain experts, not only specialists in the submethod
- produce an abstract likely to score well in the later abstract pipeline

REQUIRED CONTENT:
1. Background/problem
2. Paper-specific gap or challenge
3. What the paper does
4. Main result(s)
5. Main conclusion or implication

STRICT RULES:
- do not invent claims, numbers, baselines, or conclusions
- do not overstate novelty
- do not include implementation detail unless essential
- do not include more than 4 numeric values
- do not write in checklist tone
- do not mirror section headings
- do not say "this paper presents" unless necessary for clarity
- prefer argument flow over inventory style

METHOD ABSTRACTION RULE:
Describe what was evaluated or proposed, not low-level implementation details.

RESULT RULE:
Include the strongest supported result from the manuscript.
If quantitative evidence is available, include only the most decision-relevant numbers.

UNCERTAINTY RULE:
If the manuscript does not support a strong quantitative claim, write a conservative abstract and explicitly avoid unsupported emphasis.

OUTPUT:
--- ABSTRACT ---
<final abstract>

--- SUPPORT MAP ---
- Problem:
- Gap:
- Contribution:
- Main evidence:
- Conclusion:

--- RISKS ---
- any claim that may require manual verification
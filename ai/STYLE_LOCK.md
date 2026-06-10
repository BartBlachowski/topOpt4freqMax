SCIENTIFIC WRITING STYLE POLICY

Global priorities:
1. Preserve scientific correctness
2. Preserve readability and narrative flow
3. Apply minimal necessary changes
4. Prefer reporting issues over rewriting when uncertain

Hard constraints:
- Do not increase verbosity unless explicitly requested
- Do not convert prose into a methods checklist
- Do not add unnecessary numbers
- Do not rewrite whole sections when local edits suffice
- If a change improves rigor but harms readability, prefer reporting it
- Maintain the author’s scientific voice where possible

Narrative rule:
The manuscript should read as an argument, not an inventory.

METHOD ABSTRACTION RULE:

In abstracts:
- describe what is tested, not how it is implemented
- remove architecture-level and training-level details
- if a method term requires explanation, replace it with its purpose

ABSTRACT-SPECIFIC RULES:

- The abstract must present results at a summary level, not full experimental detail
- Maximum 4 numeric values total (unless critical)
- Do NOT report:
  - full standard deviation ranges
  - redundant derived values (Δ, absolute drop if already implied)
  - multiple equivalent expressions of the same result
- Prefer ONE clear quantitative statement per finding
- The goal is interpretability, not completeness
- Abstract sentences must express the research question, not the internal ML mechanism

NUMERIC RULE (ABSTRACT):

- Maximum: 4 numeric values
- > 4 numeric values → violation (MAJOR)
- Numbers must be non-redundant and essential to the main claim
- Do NOT report:
  - standard deviation ranges
  - duplicate representations of the same result
  - secondary/derived values (e.g., Δ if already implied)



JARGON CONTROL RULE:

- Prefer expressing the research question over internal ML mechanisms
- Replace specialized ML terms with their functional meaning where possible
- A non-AI domain expert must understand each sentence on first read
- If a sentence contains more than one specialized ML term, simplify it

PRECISION WITHOUT JARGON RULE:

- Replace vague phrases ("ways of using", "different approaches")
  with functional descriptors ("forms of information", "types of input")

- If a sentence becomes vague after removing jargon, restore precision at a higher abstraction level

ABSTRACT DISCIPLINE RULE:

- If a detail requires explanation, it does NOT belong in the abstract
- If a term is not understandable without ML knowledge, replace it
- If two numbers describe the same effect, keep only one
- The abstract reports conclusions, not experimental bookkeeping

NUMERIC PRIORITY ORDER:

1. Main result (best model performance)
2. Baseline comparison
3. Domain gap (if critical)
4. Secondary supporting number (optional)

Everything else → remove

SEVERITY DEFINITIONS:

CRITICAL:
- violates abstraction level (method details in abstract)
- introduces unreadable sentence
- changes scientific meaning
- must be fixed immediately

MAJOR:
- reduces clarity or accessibility
- includes unnecessary detail
- should be fixed

MINOR:
- stylistic improvement only
- optional

Self-check before finalizing:
- Did density increase?
- Did tone become procedural?
- Did local edits become global rewrites?
- Did readability degrade?
If yes, revise.

DO NOT:
- invent new numbers
- average results
- approximate values

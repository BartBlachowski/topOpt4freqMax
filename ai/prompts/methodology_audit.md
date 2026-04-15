---
name: methodology_audit
description: Deep technical audit of Methodology for correctness, completeness, and reproducibility
---

ROLE:
Act as a senior reviewer in computational engineering auditing the Methodology section.

INPUT:
- Use the currently focused Methodology section
- Use the project codebase (full access via VSCode)
- Use outputs of:
  - write_methodology.md
  - contribution_refiner.md

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Evaluate whether the Methodology:
- is logically correct
- is complete
- is consistent with implementation
- is reproducible
- would survive expert reviewer scrutiny

DO NOT:
- rewrite the Methodology
- improve style globally
- assume missing steps are “standard”
- ignore inconsistencies

CORE PRINCIPLE:
If a technically competent reader cannot reproduce the method from the description, the Methodology fails.

---

# AUDIT DIMENSIONS

## 1. METHOD LOGIC

Check:

- Is the method internally consistent?
- Does each step follow logically from the previous?
- Are there hidden transitions or jumps?

Detect:
- unexplained steps
- circular logic
- missing links

---

## 2. COMPLETENESS

Check whether the Methodology fully specifies:

- problem definition
- variables and notation
- objective function
- constraints
- solution procedure
- stopping criteria

Detect:
- missing definitions
- implicit assumptions
- underspecified steps

---

## 3. CODE CONSISTENCY

Compare Methodology with implementation:

For each major step:

- described in text?
- implemented in code?
- consistent?

Detect:
- described but not implemented
- implemented but not described
- mismatches in logic

---

## 4. MATHEMATICAL VALIDITY

Check:

- are equations correct?
- are variables consistently defined?
- does math correspond to actual algorithm?

Detect:
- symbolic inconsistencies
- unused equations
- misleading formalism

---

## 5. ALGORITHMIC CLARITY

Check:

- is computational flow clear?
- is iteration defined?
- is convergence defined?

Detect:
- vague algorithm description
- missing loop logic
- unclear dependencies

---

## 6. SPECIAL MECHANISMS

Focus on:

- aggregation strategies
- filtering
- stabilization
- coupling

Check:
- clearly defined?
- justified?
- correctly implemented?

These are often:
👉 the real contribution

---

## 7. PARAMETER TRANSPARENCY

Check:

- are key parameters defined?
- are their roles explained?
- are defaults or ranges given?

Detect:
- hidden hyperparameters
- unexplained constants

---

## 8. REPRODUCIBILITY

Ask:

- could an expert reimplement this from text?
- what information is missing?

List:
- missing elements required for reproduction

---

## 9. CONSISTENCY WITH CONTRIBUTIONS

Check:

- does Methodology actually implement the claimed contributions?
- or are contributions only conceptual?

Detect:
- contribution not realized in method
- mismatch between claim and implementation

---

## 10. REVIEWER ATTACK SIMULATION

Ask:

“What would a strict reviewer question here?”

Examples:
- “This step is unclear”
- “How is X computed?”
- “Where is this defined?”
- “Is this standard or new?”

---

# OUTPUT FORMAT

## --- METHODOLOGY AUDIT ---

### LOGIC
<assessment>

### COMPLETENESS
<assessment>

### CODE CONSISTENCY
<assessment>

### MATHEMATICAL VALIDITY
<assessment>

### ALGORITHMIC CLARITY
<assessment>

### SPECIAL MECHANISMS
<assessment>

### PARAMETER TRANSPARENCY
<assessment>

### REPRODUCIBILITY
<assessment>

### CONTRIBUTION CONSISTENCY
<assessment>

### REVIEWER RISK SUMMARY
<short synthesis>

---

## --- CODE ALIGNMENT MAP ---

For each major component:

- Component:
- Described in text: YES / NO
- Implemented in code: YES / NO
- Consistent: YES / PARTIAL / NO
- Issue:

---

## --- ISSUE LIST ---

For each issue:

- ID:
- Location:
- Type:
  - missing_definition
  - inconsistency
  - algorithm_gap
  - code_mismatch
  - math_issue
  - reproducibility_gap
  - parameter_missing
- Severity:
  - critical
  - major
  - minor
- Description:
- Why it matters:
- Evidence:
- Recommended action:
  - fix_now
  - fix_if_time
  - report_only
- Safe patch:
- Confidence:

---

## --- REPRODUCIBILITY GAPS ---

List explicitly:

- missing step:
- missing parameter:
- missing definition:
- missing condition:

---

## --- STRONGEST REVIEWER ATTACKS ---

1.
2.
3.
4.
5.

---

## --- METHODOLOGY VERDICT ---

Choose one:

- REPRODUCIBLE AND SOUND
- MOSTLY SOUND, MINOR GAPS
- PARTIALLY SPECIFIED
- NOT REPRODUCIBLE
- FUNDAMENTALLY UNCLEAR

Explain:
- what works
- what fails
- what must be fixed

---

## --- MINIMAL SURVIVAL PATCH ---

Provide minimal fixes:

- add:
- clarify:
- align with code:
- remove:
- do NOT change:

Rules:
- do not rewrite full section
- focus on critical gaps
- preserve valid structure

---

## --- SELF-CHECK ---

Confirm:
- I compared method against implementation
- I identified missing steps required for reproduction
- I distinguished clarity issues from real technical gaps
- I did not assume unstated knowledge